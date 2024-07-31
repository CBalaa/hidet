from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, reduce
from hidet.ir import expr

class ScatterTask(Task):
    def __init__(self, x: TensorNode, dim: int, index: TensorNode, src: TensorNode):      
        shape = x.shape
        dtype = x.type.dtype 
        reduce_shape = []
        for idx, size in enumerate(shape):
            if idx == dim:
                reduce_shape.append(size)
        
        def compute_src_element(src_idx, indices):
            indices[dim] = src_idx
            return expr.cast(src[indices], dtype)


        def y_compute(*indices):
            y_idx = indices[dim]
            def compute_src_idx(i):
                tmp = list(indices)
                tmp[dim] = i
                return expr.if_then_else(
                    cond=expr.equal(index[tmp], y_idx),
                    then_expr=i,
                    else_expr=-1
                )
            src_idx_list = compute(
                name="src_idx_list",
                shape=[index.shape[dim]],
                fcompute=compute_src_idx
            )
            src_idx = reduce(
                shape=[index.shape[dim]],
                fcompute=lambda i: src_idx_list[i],
                reduce_type='max',
                name='src_idx'
            )
            return expr.if_then_else(
                cond = expr.equal(src_idx, -1),
                then_expr = x[indices],
                else_expr = compute_src_element(src_idx, list(indices))
            )
        
        y = compute(
            name = "y",
            shape = shape,
            fcompute = y_compute
        )
        super().__init__(name='scatter', inputs=[x, index, src], outputs=[y], attributes={'dim': dim})
    

class ScatterOp(Operator):
    def __init__(self, x: Tensor, dim: int, index: Tensor, src: Tensor):
        super().__init__(
            inputs=[x, index, src], 
            attributes={'dim': dim}, 
            task=ScatterTask(input_like(x, 'x'), dim, input_like(index, 'index'), input_like(src, 'src'))
        )

def scatter(x, dim, index, src):
    return ScatterOp(x, dim, index, src).outputs[0]
