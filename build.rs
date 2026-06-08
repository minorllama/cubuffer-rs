// build.rs
fn main() {
    // Compile the CUDA kernel
    println!("cargo:warning=cargo/nvcc...");
    std::process::Command::new("nvcc")
        .args(&["-O3", "-c", "src/kernel.cu", "-o", "kernel.o", "-Xcompiler", "-fPIC"])
        .status()
        .unwrap();

    // Archive into a static library
    std::process::Command::new("ar")
        .args(&["rcs", "libcubuffer.a", "kernel.o"])
        .status()
        .unwrap();

    // Tell cargo where to find the library and what to link
    println!("cargo:rustc-link-search=native=.");
    println!("cargo:rustc-link-lib=static=cubuffer");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    
    // Rebuild if the kernel changes
    println!("cargo:rerun-if-changed=src/kernel.cu");
}
