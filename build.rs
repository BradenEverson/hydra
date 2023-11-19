extern crate cc;

fn main(){
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .file("src/cuda/matrix.cu")
        .compile("libmatrix_math.a");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

}
