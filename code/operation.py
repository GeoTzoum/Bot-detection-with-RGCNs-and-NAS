import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
 
##
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param

import torch_geometric.typing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.utils import index_sort, one_hot, scatter, spmm
from torch_geometric.utils.sparse import index2ptr


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGCNConvTP(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    .. note::
        :class:`RGCNConv` can use `dynamic shapes
        <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index
        .html#work_dynamic_shapes>`_, which means that the shape of the interim
        tensors can be determined at runtime.
        If your device doesn't support dynamic shapes, use
        :class:`FastRGCNConv` instead.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        num_blocks (int, optional): If set, this layer will use the
            block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by :obj:`edge_type`. This avoids
            internal re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'mean',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        trasform_or_propagate="P",
        edge_index=None,
        edge_relations=None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted
        self.use_segmm: int = -1
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]
        self.trasform_or_propagate = trasform_or_propagate
        
        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.Tensor(num_relations, num_blocks,
                             in_channels[0] // num_blocks,
                             out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.edge_index = edge_index
        self.edge_relations = edge_relations
        
    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)


    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index=None, edge_type: OptTensor = None):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or tuple, optional): The input node features.
                Can be either a :obj:`[num_nodes, in_channels]` node feature
                matrix, or an optional one-dimensional node index tensor (in
                which case input features are treated as trainable node
                embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_type (torch.Tensor, optional): The one-dimensional relation
                type/index for each edge in :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.SparseTensor`. (default: :obj:`None`)
        """
        #x has shape (num_relations, num_nodes, in_channels)
        # if x has not n_relations in the first dimension, then copy x for each relation
        edge_index, edge_type = self.edge_index, self.edge_relations
        
        if x.size(0) != self.num_relations:
            x = x.repeat(self.num_relations+1, 1, 1)

        size = (x[0].size(0), x[0].size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x[0].size(0), self.out_channels, device=x[0].device)

        weight = self.weight
        
        if(self.trasform_or_propagate=="T"):#only transform
            out = None
            for i in range(self.num_relations):
                #multiply features with weight for each relation and store them in each row of out
                res = x[i] @ weight[i]
                if(out is None):
                    out = res.unsqueeze(0)
                else:
                    out = torch.cat((out,res.unsqueeze(0)),dim=0)
                    
            root = self.root
            out = torch.cat((out,(x[-1] @ root).unsqueeze(0)),dim=0)

            if self.bias is not None:
                out = out + self.bias
            #add activation function
            out = F.relu(out)
            return out #shape (num_relations+1, num_nodes, out_channels)
        else:#only propagate
            assert edge_type is not None
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x[i], edge_type_ptr=None,
                                    size=size)
                out = out + h
            #add central node features
            out = out + x[-1]
            return out #shape (num_nodes, out_channels)

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if torch_geometric.typing.WITH_PYG_LIB and edge_type_ptr is not None:
            # TODO Re-weight according to edge type degree for `aggr=mean`.
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')


class Graph(nn.Module):
    def __init__(self, adj):
        super(Graph, self).__init__()
        self.adj = adj

    def forward(self, x):
        x = self.adj.matmul(x)
        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nclass, dropout, last=False):
        super(MLP, self).__init__()
        self.lr1 = nn.Linear(nfeat, nclass)
        self.dropout = dropout
        self.last = last

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr1(x)
        if not self.last:
            x = F.relu(x)
        return x

class ModelOp(nn.Module):
    def __init__(self, arch, hid_dim, num_relations, num_classes, fdropout, dropout, edge_index,edge_relations,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3):
        super(ModelOp, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        edge_index = edge_index.cuda()
        edge_relations = edge_relations.cuda()
        for element in arch:
            if element == 1:
                #op = Graph(adj)
                op = RGCNConvTP(hid_dim, hid_dim, num_relations, edge_index=edge_index, edge_relations=edge_relations,trasform_or_propagate="P")
                self._numP += 1
            elif element == 0:
                #op = MLP(hid_dim, hid_dim, dropout)
                op = RGCNConvTP(hid_dim, hid_dim, num_relations, trasform_or_propagate="T")
            else:
                print("arch element error")
            self._ops.append(op)
        self.gate = torch.nn.Parameter(1e-5*torch.randn(self._numP), requires_grad=True)
        #self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)
        self.num_relations = num_relations
        
        #botrgcn
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(hid_dim/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(hid_dim/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(hid_dim/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(hid_dim/4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input=nn.Sequential(
            nn.Linear(hid_dim,hid_dim),
            nn.LeakyReLU()
        )
        
        
    def forward(self, des,tweet,num_prop,cat_prop):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT  = []
        for i in range(len(self._arch)):
            if i == 0:
                d = self.linear_relu_des(des)
                t = self.linear_relu_tweet(tweet)
                n = self.linear_relu_num_prop(num_prop)
                c = self.linear_relu_cat_prop(cat_prop)
                res = torch.cat((d,t,n,c),dim=1)
                res = self.linear_relu_input(res)

                tempP.append(res)
                numP.append(i)
                totalP += 1
                
                res = self._ops[i](res)
                if self._arch[i] == 1:
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    tempT.append(res)
                    numP = []
                    tempP = []
            else:
                if self._arch[i - 1] == 1:
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        numP.append(i - point)
                        totalP += 1
                else:
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)
        if len(numP) > 0 or len(tempP) > 0:
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        #if res has 3 dimensions, then take the mean of the first dimension
        if len(res.shape)==3:
            res = torch.mean(res,dim=0)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits