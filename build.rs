extern crate cc;

fn main(){
    cc::Build::new()
        .cuda(true)
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .file("src/cuda/matrix.cu")
        .compile("libmatrix_math.a");
}
