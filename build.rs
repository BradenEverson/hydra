extern crate cc;

fn main(){

    cc::Build::new()
        .cuda(true)
        .file("src/cuda/matrix.cu")
        .flag("-O2")
        .compile("libmatrix_math.a");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

}
