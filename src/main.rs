use cubuffer::cubufferapi::*;

fn main() {
    let size:usize = 8;
    let host_buffer_a:Vec<f32> = vec![1 as f32; size]; 
    println!("{:?}", host_buffer_a);
    
    let mut buffer = CuBuffer::<f32>::new(size);
    buffer.from_host(&host_buffer_a);
    cubuffer_test(&mut buffer, size);

    let mut host_buffer_b:Vec<f32> = vec![0 as f32; size];
    buffer.to_host(&mut host_buffer_b);
    eprintln!("{:?}", host_buffer_b); 
}


