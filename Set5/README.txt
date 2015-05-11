What is the latency of classifying a single batch? How does this change with
batch size?

  Clusters: 50 Batch_size: 1024
  
  Host to Device: 0.106496 ms / batch
  Kernel: 1.262848 ms / batch
  Device To Host: 0.016128 ms / batch
  
  Clusters: 50, Batch_size: 2048
  
  Host to Device: 0.207456 ms / batch
  Kernel: 1.616544 ms / batch
  Device To Host: 0.022464 ms / batch
  
  Clusters: 50, Batch_size: 4096
  
  Host to Device: 0.397568 ms / batch
  Kernel: 2.262592 ms / batch
  Device To Host: 0.028320 ms / batch

  The latency above is split into finding the time for Host to Device, Kernel,
  and then copying the output back to device. Host to Device times are a magnitude
  larger than the device to host times because we copy REVIEW_DIM * batch_size
  into device for Host to device and we copy batch_size for device to host. Note
  that the host to device and device to host increases linearly with batchsize,
  because we are copying more data. Note that kernel also increases in time
  because for each point in batch_size of input we calculate the min distances
  and updates the minimum distanced cluster.

What is the throughput of cluster.cc in reviews / s? How does this change with
batch size? How does this throughput compare to that of loader.py? Considering
these throughputs, is there any advantage to running the cluster program
on pre-computed LSA's (using cat) rather than piping in the output of
loader.py?

  We have ms / batch. (ms / batch)^-1 * (reviews / batch) * (1000 ms/ s) = review/sec
  
  Stage Throughputs: 
  
  Clusters: 50 Batch_size: 1024
  
  Host to Device: 9615384.615 reviews / s
  Kernel: 810865.599 reviews / s
  Device To Host: 63492063.49 reviews / s
  
  Clusters: 50, Batch_size: 2048
  
  Host to Device: 9871972.852 reviews / s
  Kernel: 1266900.251 reviews / s
  Device To Host: 91168091.17 reviews / s
  
  Clusters: 50, Batch_size: 4096
  
  Host to Device: 10302640.05 reviews / s
  Kernel: 1810313.128 reviews / s
  Device To Host: 144632768.4 reviews / s
  
  Pipeline Throughput (minimum of stage throughputs):
  
  Clusters: 50, Batch_size: 1024 : 810865.599 reviews / s
  Clusters: 50, Batch_size: 2048 : 1266900.251 reviews / s
  Clusters: 50, Batch_size: 4096 : 1810313.128 reviews / s
  
  We see that the throughput of the cluster.cc increases as the batch sizes
  goes up. This means that we are processing reviews at a faster rate. This is
  because we are doing more work at each iteration. If we use loader.py, the
  throughput will increase because the bottleneck is actually the loading
  the data into the program. The actual stage throughput of copying host to device,
  kernel and device to host does not change much. The advantage is that we 
  do not have to compute the LSA reviews from json and it is faster to pipe in 
  the data

Does cluster saturate the PCI-E interface between device and host? You might
want to use the program available at
https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/
optimize-data-transfers/bandwidthtest.cu
to check the bandwidth from host to device. For what value of k does the
bottleneck switch between host-device IO and the kernel?


  Device: GeForce GTX 570
  Transfer size (MB): 16
  
  Pageable transfers
    Host to Device bandwidth (GB/s): 2.288118
    Device to Host bandwidth (GB/s): 3.216550
  
  Pinned transfers
    Host to Device bandwidth (GB/s): 3.297616
    Device to Host bandwidth (GB/s): 3.346640
  
  The bandwidth from host to device is:
  1923 MB /s for clusters = 50 and batch size = 1024
  
  The bandwidth is around 2 GB / s, which means that the cluster does not
  saturate the PCI-E interface.
  
  I played around with k to try to figure out when the kernel would have a 
  latency low enough that it goes below the latency for host device IO,
  so that the bottleneck would switch. However, I was unsuccessful in doing so.
  In theory, if I increase the batch_size, so that I would be increasing the 
  amount of data transferred between device and host to increase the 
  device IO latency. In addition, I decreased k, which is the number of cluster
  centers. With less k, there will be less cluster centers to check
  through and thus the kernel will take less time. However, if you decrease
  k to a much lower number, you see the latency to increase because we are using
  atomic adds, so as k decreases, theres more chance theres blocking because
  we are incrementing the same cluster count. 

Do you think you could improve performance using multiple GPUs? If so, how?
If not, why not?

  We could improve performance using multiple GPUs. To do this we will take 
  advantage of data parallelism. This will effectively remove throughput
  bottlenecks. In our code we will split data amongst all our GPUs. One of the
  GPUs can hold the data with the cluster center locations and counts, and the
  other GPUS can access their peer GPU and the cluster data that they need
  to update. This means that we can speed up the computation even more by
  parallelizing it even more between GPUs.  