#include <iostream>
#include <list>

#include <thrust/host_vector.h>   //stl
#include <thrust/device_vector.h>

#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/functional.h>    //std::algorithm
#include <thrust/iterator/zip_iterator.h>

using std::cout;
using std::endl;

//仿函数 实现y = a * x + y
class Saxpy {
 public:
  Saxpy(float a)
    : a_(a) {}
  __host__ __device__ float operator()(float x, float y) const {
    return a_ * x + y;
  }
  
 private:
  const float a_;
};

//仿函数 实现y = a * a
template <typename T>
class Square {
 public:
  __host__ __device__ T operator()(T x) const {
    return x * x;
  }
};


void saxpy_fast(float a, 
                thrust::device_vector<float>& x,
                thrust::device_vector<float>& y) {
  //二元操作 第三个参数是传入函数的第二个实参 每运行一次 迭代器移动一次 
  //又和结果参数是in place 所以每运行一次 替换一个y的值
  thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), Saxpy(a));
}

int main() {
  thrust::host_vector<int> a(4);
  a.resize(10);
  //全部填充6
  thrust::fill(a.begin(), a.end(), 6);
  for (int i = 0; i < a.size(); i++) {
    cout << a[i] << "\t";
  }
  cout << endl;
  
  thrust::device_vector<int> b(10);
  //值为0 1 2 3...
  thrust::sequence(b.begin(), b.end());
  for (int i = 0; i < b.size(); i++) {
    cout << b[i] << "\t";
  }
  cout << endl;
  
  //copy a to b
  thrust::copy(a.begin(), a.end(), b.begin());
  for (int i = 0; i < b.size(); i++) {
    cout << b[i] << "\t";
  }
  cout << endl;

  //使用原始指针
  int* raw_ptr = nullptr;
  cudaMalloc((void**)&raw_ptr, sizeof(int) * 10);
  //用device_ptr包装原始指针
  thrust::device_ptr<int> dev_ptr(raw_ptr);
  thrust::fill(dev_ptr, dev_ptr + 10, 7);
  for (int i = 0; i < 10; i++) {
    cout << dev_ptr[i] << "\t";
  }
  cout << endl;
  //device_ptr中提取原始指针
  raw_ptr = thrust::raw_pointer_cast(dev_ptr);
  
  //迭代器可以遍历多种数据结构
  std::list<int> host_list;
  host_list.push_back(10);
  host_list.push_back(20);
  host_list.push_back(30);
  host_list.push_back(40);
  
  thrust::copy(host_list.begin(), host_list.end(), b.begin());
  for (int i = 0; i < b.size(); i++) {
    cout << b[i] << "\t";
  }
  cout << endl;

  //转换(transform/for_each) 应用于输入范围中的每个元素
  thrust::device_vector<int> c(b.size());
  //std::negate<int>() 是仿函数 取参数相反
  thrust::transform(b.begin(), b.end(), c.begin(), thrust::negate<int>());
  for (int i = 0; i < c.size(); i++) {
    cout << c[i] << "\t";
  }
  cout << endl;

  //替换
  thrust::replace(c.begin(), c.end(), -6, 8);
  for (int i = 0; i < c.size(); i++) {
    cout << c[i] << "\t";
  }
  cout << endl;
  
  //看看有多少个8
  int count = thrust::count(c.begin(), c.end(), 8);
  cout << count << endl;
  

  //blas中的saxpy算法
  thrust::device_vector<float> aa(5);
  thrust::sequence(aa.begin(), aa.end());
  thrust::device_vector<float> bb(5);
  thrust::sequence(bb.begin(), bb.end());
  for (int i = 0; i < aa.size(); i++) {
    cout << aa[i] << "\t";
  }
  cout << endl;

  saxpy_fast(5, aa, bb);
  for (int i = 0; i < bb.size(); i++) {
    cout << bb[i] << "\t";
  }
  cout << endl;

  //reduce 每次调用函数得到结果 再和下一个元素继续调用
  float sum = thrust::reduce(aa.begin(), aa.end(), 0);
  cout << sum << endl;

  float init = 0;
  //transform_reduce 可以实现inner_product 向量内积 第一个op是相加 第二个op相乘
  //thrust::transform_reduce(aa.begin(), aa.end(), bb.begin(), 0, thrust::plus<float>(), thrust::multiplies<float>());
  //先用transform算每个值的平方值 再用reduce算平方和 最后开方
  float norm = std::sqrt(thrust::transform_reduce(aa.begin(), aa.end(), Square<float>(), init, thrust::plus<float>()));
  cout << norm << endl;

  //特殊迭代器 zip_iterator可以把多个序列 合成一个元祖序列
  thrust::device_vector<int> d1(5);
  thrust::sequence(d1.begin(), d1.end());
  thrust::device_vector<char> d2(5);
  d2[0] = 'a';
  d2[1] = 'b';
  d2[2] = 'c';
  d2[3] = 'd';
  d2[4] = 'e';

  auto first = thrust::make_zip_iterator(thrust::make_tuple(d1.begin(), d2.begin()));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(d1.end(), d2.end()));

  //创建迭代器 得到最大值 
  thrust::maximum<thrust::tuple<int, char>> binary_op;
  thrust::tuple<int, char> init_ = thrust::make_tuple(10, 'h');
  thrust::reduce(first, last, init_, binary_op);

  return 0;
}
