# Path Forward Beyond Simulators: Fast and Accurate GPU Execution Time Prediction for DNN Workloads

**Ying Li**  
William & Mary  
Williamsburg, VA, USA  
yli81@wm.edu  

**Yifan Sun**  
William & Mary  
Williamsburg, VA, USA  
ysun25@wm.edu  

**Adwait Jog**  
University of Virginia  
Charlottesville, VA, USA  
ajog@virginia.edu  

## ABSTRACT
Today, DNNs’ high computational complexity and sub-optimal device utilization present a major roadblock to democratizing DNNs. To reduce the execution time and improve device utilization, researchers have been proposing new system design solutions, which require performance models (especially GPU models) to help them with pre-product concept validation. Currently, researchers have been utilizing simulators to predict execution time, which provides high flexibility and acceptable accuracy, but at the cost of a long simulation time. Simulators are becoming increasingly impractical to model today’s large-scale systems and DNNs, urging us to find alternative lightweight solutions.

To solve this problem, we propose using a data-driven method for modeling DNNs system performance. We first build a dataset that includes the execution time of numerous networks/layers/kernels. After identifying the relationships of directly known information (e.g., network structure, hardware theoretical computing capabilities), we choose to build a simple, yet accurate, performance model for DNNs execution time. Our observations on the dataset demonstrate prevalent linear relationships between the GPU Kernel execution time, operations count, and input/output parameters of DNNs layers. Guided by our observations, we develop a fast, linear-regression-based DNNs execution time predictor. Our evaluation using various image classification models suggests our method can predict new DNNs performance with a 7% error and new GPT performance with a 15.2% error. Our case studies also demonstrate how the performance model can facilitate future DNNs system research.

## CCS CONCEPTS

- Computing methodologies → Modeling methodologies.

## KEYWORDS
Deep Neural Networks; Graphics Processing Units; Performance Model

## ACM Reference Format:
Ying Li, Yifan Sun, and Adwait Jog. 2023. Path Forward Beyond Simulators: Fast and Accurate GPU Execution Time Prediction for DNN Workloads. In *56th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO '23), October 28–November 01, 2023, Toronto, ON, Canada.* ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3614324.3614277

---

This work is licensed under a Creative Commons Attribution-NoDerivs International 4.0 License.

MICRO '23, October 28–November 01, 2023, Toronto, ON, Canada  
© 2023 Copyright held by the owner/author(s).  
ACM ISBN 978-1-4503-9742-3/23/10.  
https://doi.org/10.1145/3614324.3614277

---

## 1 INTRODUCTION
Deep neural networks (DNNs) are becoming increasingly popular because they have the extraordinary capacity of performing tasks that typically require significant human involvement (e.g., recognizing objects in images [26, 61, 63, 73], processing natural languages [4, 52, 71]). DNNs' power leads to the proliferation of DNNs, as demonstrated by a large number of DNNs available on the Huggingface [70] platform designed to solve various problems. Today, most DNNs consume a huge amount of computing power [62], preventing practitioners from lowering DNNs deployment costs [13, 16, 75] and boarding the user bases [10].

Improving the efficiency of DNNs requires better system (broadly defined as the collection of software, run-time library, operating system, and hardware) designs. As a critical process of developing new solutions, researchers typically need to evaluate the performance [24] of the emerging systems and compare them with a baseline design. Since building new systems (making changes to hardware, operating system, or machine learning software) is costly [13, 45], researchers commonly use performance modeling methods to predict the performance and validate their design ideas. Moreover, since DNN workloads are mainly executed on Graphic Processing Units (GPUs), and GPUs are likely to be a performance bottleneck [29], it is critical to provide a high-performance, high-flexibility, and high-accuracy performance model for DNNs running on GPUs.

Researchers have been developing simulators to predict DNNs performance [5, 21, 72]. While GPU simulators provide great flexibility and a reasonable error around 20% to 70% [2, 9, 41, 42, 64, 69], the long simulation time hinders researchers from evaluating large-scale and complex systems executing long-running applications. For example, it is reported that GPGPU-Sim [7] may need years to centuries [5] to simulate end-to-end machine learning workloads. Modern system-level research inevitably involves large DNNs (e.g., GPT [56]) that execute on large systems (e.g., multi-GPU systems), pushing researchers to stay away from simulators and rely on real systems. However, using real systems has disadvantages, including 1) high acquiring costs and 2) not being able to evaluate non-existing hardware. Therefore, the community urgently demands a new set of solutions that accurately model DNNs system performance, which requires approaching the problem from a new direction.

To address the aforementioned problem, we take a data-driven approach that is complimentary to simulator development. Our goal is to develop a faster performance model that can provide a similar or even more accurate estimation of the DNNs performance on GPUs compared to simulators. To achieve this goal, we first collect a large number (464) of models from commonly used model zoos (e.g., HuggingFace [70], TorchVision [43]) and measure the
