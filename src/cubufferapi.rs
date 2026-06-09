unsafe extern "C" {
    fn cuda_malloc(size: usize) -> *mut std::ffi::c_void;
    fn cuda_free(ptr: *mut std::ffi::c_void);
    fn cuda_copy_to_host(host_ptr:*mut std::ffi::c_void, device_ptr:*const std::ffi::c_void, n:usize)-> i32;
    fn cuda_copy_to_device(device_ptr:*mut std::ffi::c_void, host_ptr:*const std::ffi::c_void, n:usize)-> i32;
}

unsafe extern "C" {
    fn cuda_test_alloc_f32(ptr: *mut f32, n:usize);
    fn cuda_test_alloc_i32(ptr: *mut f32, n:usize);
    fn cuda_example_vadd(a: *const f32, b: *const f32, c: *mut f32, n: i32);
    fn cuda_test_thread_id_f32(ptr: *mut f32, n:usize);
}


pub struct CuBuffer<T> {
    ptr: *mut T,
    size: usize,
}

const LOG_LEVEL:i32 = 0; // to_do --cfg instead

impl<T: Copy> CuBuffer<T> {
    pub fn new(size: usize) -> Self {
        let n_bytes = size * size_of::<T>();
        let raw_ptr = unsafe { cuda_malloc(n_bytes) };
        assert!(!raw_ptr.is_null(), "CUDA allocation failed!");
        
        Self { 
            ptr: raw_ptr as *mut T, 
            size 
        }
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn n_bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }

    pub fn to_host(self, v:&mut Vec<T>) {
        assert!(
            v.capacity() >= self.size,
            "Target vector capacity ({}) is smaller than CuBuffer size ({})",
            v.capacity(),
            self.size
        );

        unsafe {
            let host_ptr = v.as_mut_ptr() as *mut std::ffi::c_void;
            let device_ptr = self.ptr as *const std::ffi::c_void;

            let bytes_to_copy = self.n_bytes();

            let status = cuda_copy_to_host(host_ptr, device_ptr, bytes_to_copy);
            assert_eq!(status, 0, "cuda_copy_to_host failed with status: {}", status);

            // set length; ok since capacity at at least length
            if v.len() < self.size {
                v.set_len(self.size);
            }
        }
    }

    pub fn from_host(&mut self, v:&Vec<T>) {
        assert!(
            v.len() <= self.size,
            "Target vector capacity ({}) is smaller than CuBuffer size ({})",
            v.len(),
            self.size
        );

        unsafe {
            let host_ptr = v.as_ptr() as *const std::ffi::c_void;
            let device_ptr = self.ptr as *mut std::ffi::c_void;

            let bytes_to_copy = v.len() * std::mem::size_of::<T>();

            let status = cuda_copy_to_device(device_ptr, host_ptr, bytes_to_copy);
            assert_eq!(status, 0, "cuda_copy_to_device failed with status: {}", status);
        }
    }

}

impl<T> Drop for CuBuffer<T>{
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.ptr as *mut std::ffi::c_void);
        }
        if LOG_LEVEL > 0 { println!("freed"); }
    }
}


pub fn cubuffer_test(buffer: &mut CuBuffer<f32>, size:usize){
    unsafe { cuda_test_thread_id_f32(buffer.as_mut_ptr(), size); };
}

pub fn cubuffer_f32_copy_test() {
    let size:usize = 8;
    let host_buffer_a:Vec<f32> = vec![1 as f32; size]; 
    println!("{:?}", host_buffer_a);
    
    let mut buffer = CuBuffer::<f32>::new(size);
    buffer.from_host(&host_buffer_a);
    //unsafe { cuda_test_thread_id_f32(buffer.as_mut_ptr(), size); };
    cubuffer_test(&mut buffer, size);

    let mut host_buffer_b:Vec<f32> = vec![0 as f32; size];
    buffer.to_host(&mut host_buffer_b);
    eprintln!("{:?}", host_buffer_b); 
}




