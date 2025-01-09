use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    // 只有1次循环
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        // 句子的长度offset
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            // 在哪一行的那一列offset
            let offset = base + i * total_seq_len;
            // +1是倒不了
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));
            // 闭包查找最大值,当前值a和下一个值b比较，返回最大值
            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();
                // 生成的e求和
            (0..boundary).for_each(|j| data[offset + j] /= sum);
            // 后面的全部看不到的位置填充0
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let len = y.size();
    assert!(len == x.size());
    let n_col = y.shape()[y.shape().len()-1];
    let n_row = y.shape()[y.shape().len()-2];
    let batch = y.size() / (n_col * n_row);
    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    // 获取多少个列向量
    
    // 计算每一个列项的长度,直接使用w
    // 计算多少个block,对每一个block里面的列向量来增加
    // for b in (0..batch){
    //     let base = b * n_col * n_row;
    //     let mut rec = vec![0.0 as f32;n_col];
    //     for i in (0..n_col*n_row){
    //         // 获取当前的x
    //         // _y[i+base] = _x[i+base]*w.data()[(i+base)%n_col];
    //         // 统计x[i]
           
    //         rec[i%n_col]+= _x[i+base]*_x[i+base];
    //     }
    //     rec.iter_mut().for_each(|j|*j =(*j/n_row as f32).sqrt()+epsilon);
    //     for i in (0..n_col*n_row){
    //         // 获取当前的x
    //         _y[i+base] = _x[i+base]*w.data()[(i)%n_col]/rec[i%n_col];
    //         // 统计x[i]
           
    //     }
    // }
    // 预分配 rec 向量，并在每个批次前重置
  
    let mut rec = vec![0.0f32; n_row];
    for b in 0..batch {
        let base = b * n_col * n_row;

        // 重置 rec 向量
        for j in 0..n_row {
            rec[j] = 0.0;
        }

        // 计算每一行的平方和
        for i in 0..(n_col * n_row) {
            let row_idx = i / n_col;
            rec[row_idx] += _x[i + base] * _x[i + base];
        }

        // 计算 RMS 并添加 epsilon
        for j in 0..n_row {
            rec[j] = (rec[j] / n_col as f32).sqrt() + epsilon;
        }

        // 归一化并应用权重
        for i in 0..(n_col * n_row) {
            let row_idx = i / n_col;
            let col_idx = i % n_col;
            _y[i + base] = _x[i + base] * w.data()[col_idx] / rec[row_idx];
        }
    }
      
        // 首先计算输出有多少个block
        // 之后对每一个block来开始进行按照+col的来计算
        // 得到最后的长度
    

    
    
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let shape_x = x.shape();
    // step1:获取整个长度进行sigmoid
    let x_len = x.size();
    let mut out = Tensor::new(vec![0.0 as f32;x_len],shape_x);
    let _out =unsafe { out.data_mut() };
    // (0..x_len).for_each(|j|_out[j]=1.0/(1.0+(-_x[j]).exp()));
    
    // (0..x_len).for_each(|j|_out[j]*=_x[j]);
    // (0..x_len).for_each(|j|_y[j]*=_out[j]);
    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
    // 优化到一个
    for (y_it,x_it) in _y.iter_mut().zip(_x){
        let sig = 1.0 / (1.0 + (-x_it).exp());
        let six = *x_it *sig;
        *y_it*=six;

    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let a_row = a.shape()[a.shape().len()-2];
    let a_col = a.shape()[a.shape().len()-1];
    let b_row = b.shape()[b.shape().len()-2];
    let b_col = b.shape()[b.shape().len()-1];
    let c_row = c.shape()[c.shape().len()-2];
    let c_col = c.shape()[c.shape().len()-1];
    assert!(a_col == b_col);
    assert!(a_row == c_row);
    assert!(b_row == c_col);
    // 计算batch
    let batch = c.size()/(c_row * c_col);
    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();
    let a_batch = a.size()/(a_row * a_row);
    let b_batch = b.size()/(b_row * b_row);
    // assert!(a_batch == b_batch);
    let _ = _c.iter().map(|j|beta*j);
    for b in (0..batch) {
        let c_offset = b * c_row * c_col;
        let a_offset = b*a_row * a_col;
        let b_offset = b*b_row * b_col;
        for i in 0..c_row {
            for j in 0..c_col {
                // _c[i*c_col+j+c_offset]+=
            //     获取a，b来进行点击
                let ax = Tensor::new(
                    _a[(a_offset+i*a_col)..(a_offset+(i+1)*a_col)].to_vec(),
                    &vec![1,a_col]
                );
                let by = Tensor::new(
                    _b[(b_offset+j*b_col)..(b_offset+(j+1)*b_col)].to_vec(),
                    &vec![1,b_col]
                );
                let plus = dot(&ax,&by)*alpha;
                _c[i*c_col+j+c_offset]+=plus;
            }
        }
    }


}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    // 句子长度为 1，词向量长度为 3
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    println!("{:?}", y);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
