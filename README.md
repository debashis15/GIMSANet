# GIMSANet: Gradient-Infused Multi-Scale Attention Network for Robust Image Denoising

Welcome to the official repository for **GIMSANet**, a cutting-edge deep learning framework for robust image denoising, addressing the challenges of both synthetic and real-world noisy scenarios.

---

## Abstract
In the realm of digital image processing, the removal of noise from images remains a crucial preprocessing step for tasks such as object detection, segmentation, and tracking. Convolutional Neural Networks (CNNs) have become popular tools for image denoising due to their flexible architecture. However, despite advancements in deep learning, there are still some fundamental limitations. 

For instance:
- The importance of **gradient information** in the denoising process is often overlooked.
- Conventional CNNs struggle to handle **diverse noise levels and distributions** found in real-world images, limiting their effectiveness in producing high-quality denoised outputs under various conditions.

Our research highlights the detrimental effects of ignoring gradient information in many real-world scenarios, leading to a significant decrease in denoising performance. 

To address these challenges, we propose the **Gradient-Infused Multi-Scale Attention Driven CNN for Robust Image Denoising (GIMSANet)**. This novel denoising model is designed to handle both **synthetic additive white Gaussian noise (AWGN)** and **real image denoising tasks**. Through extensive experimentation, we demonstrate that GIMSANet surpasses existing methods. Furthermore, comprehensive ablation studies confirm the effectiveness of our approach.

---

## Publication
This work has been **accepted for publication at INDICON 2024**, the flagship annual international conference of the **IEEE India Council**.

### About INDICON
INDICON is the most prestigious annual conference of IEEE India Council, focusing on the fields of Computer Science and Engineering, Electrical Engineering, and Electronics and Communication Engineering. 

- INDICON has evolved from the Annual Convention and Exhibitions (ACE), restructured in 2003 to better serve the IEEE community. 
- The inaugural INDICON was hosted by the **IEEE Kharagpur Section in 2004**, and after two decades, INDICON 2024 will once again be hosted by the same section.

We are honored to contribute to the legacy of this esteemed conference.

---

## Features
- **Gradient Awareness**: Incorporates gradient information to enhance detail preservation.
- **Multi-Scale Attention Mechanisms**: Captures and integrates features across different scales.
- Handles both **synthetic AWGN** and **real-world noisy images**, making it highly versatile.
- **Extensive Ablation Studies**: Validates the contributions of individual components within the network.

---

## Datasets Used

### **Synthetic Grayscale Images**
- **Training Dataset**: BSD400  
- **Testing Dataset**: BSD68  

### **Synthetic Color Images**
- **Training Dataset**: CBSD432  
- **Testing Dataset**: BSD68  

### **Real-World Noisy Images**
- **Training Dataset**: SIDD (Smartphone Image Denoising Dataset) training set.  
- **Testing Datasets**:
  - **SIDD Validation Set**  
  - **PolyU Dataset**  
  - **NAM Dataset**  

---

## Code and Data
The **source code**, **pre-trained models**, and additional datasets will be made available soon. Please watch this repository for updates.

---

## Results and Metrics
For detailed **quantitative results**, **performance metrics**, or **visual comparisons**, feel free to contact the corresponding author:

- **Debashis Das**
  - Email 1: [ddebashisdas2108@gmail.com](mailto:ddebashisdas2108@gmail.com)
  - Email 2: [debashis_2221cs31@iitp.ac.in](mailto:debashis_2221cs31@iitp.ac.in)

---

## Citation
If you find this repository useful in your research, please consider citing our paper. Citation details will be provided once INDICON 2024 proceedings are published.

---

## Contact
For any queries, feedback, or collaboration opportunities, please reach out to **Debashis Das** at the emails provided above.

---

Thank you for your interest in GIMSANet!
