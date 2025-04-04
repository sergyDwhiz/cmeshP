# Pseudo Rendering: A Novel Deep Learning Approach for 2D Cortical Mesh Segmentation

Complete Article: [Pseudo Rendering: A Novel Deep Learning Approach for 2D Cortical Mesh Segmentation (MIT SGI 2024)](https://summergeometry.org/sgi2024/pseudo-rendering-a-novel-deep-learning-approach-for-2d-cortical-mesh-segmentation/)

# ![Project Cover Image](<Pic 1 - Mesh.png>)

## Abstract
This project introduces a novel methodology for segmenting cortical meshes using 2D projection descriptors combined with advanced deep learning and unsupervised clustering techniques. By transforming complex three-dimensional brain surface data into multiple two-dimensional views, our approach circumvents traditional computational constraints while offering a scalable, robust segmentation pipeline. We detail the entire process—from data acquisition and advanced preprocessing to deep convolutional network design and multi-view integration—culminating in precise 3D reconstruction of cortical structures.

## Introduction
Brain surface segmentation is a pivotal task in neuroimaging, essential for accurate diagnosis and treatment planning in numerous neurological disorders. Traditional segmentation methods often struggle with the complexity and variability inherent in cortical anatomy. In this work, I propose an innovative approach that:
- Transforms 3D cortical meshes into multiple informative 2D projections.
- Leverages state-of-the-art deep convolutional neural networks for high-resolution segmentation.
- Reconstructs the segmented 2D maps back to a detailed 3D anatomical representation.

This method significantly reduces computational overhead while retaining critical structural information, paving the way for advanced neuroimaging analyses.

## Literature Review
While volumetric segmentation using 3D CNNs has been extensively studied, converting 3D geometry into 2D representations remains underexplored. Prior works [Author et al., Year; Another Author et al., Year] have demonstrated benefits in multi-view processing, yet often lack robust methods for consistent 3D reconstruction. My approach builds on these foundational studies by integrating multi-view deep learning with rigorous post-processing to produce reliable, high-fidelity cortical segmentations.

## Methodology
### Multi-view Projection Extraction
- **View Selection & Rendering:** Use pyrender and trimesh to generate diverse perspectives of cortical meshes, ensuring six canocical complementary viewpoints (Camera views)
![alt text](image.png)
- Add ground truth (annotations)
![alt text](image-1.png)
- **Projection Descriptor Extraction:** Convert 3D anatomical structures into high-fidelity 2D images with minimal loss of geometric features.
- **Add annotations and Curvature:** To continue, we take the 2D projections obtained from the six camera views and perform annotations and curvature calculations, which are essential for understanding the cortical surface’s geometry and features.
![alt text](image-2.png)

### Segmentation Model
- **Deep CNN Architecture:** Employ a custom network (e.g., CorticalSegmentationCNN) to process the 2D projections, yielding pixel-wise segmentation maps.
- **Unsupervised Clustering:** Apply K-means clustering to refine segmentation boundaries based on vertex feature distributions.

### 3D Reconstruction
- **Inverse Mapping:** Integrate multiple 2D segmentation outputs via transformation matrices to reconstruct the original 3D cortical mesh.
- **Evaluation Metrics:** Quantify performance using Mean Squared Error (MSE), accuracy, and curvature consistency to validate segmentation fidelity.

## Experiments & Results
### Dataset & Preprocessing
- **Data Sources:** Cortical surfaces generated via FreeSurfer from real-world neuroimaging studies.
- **Preprocessing:** Includes normalization, vertex realignment, and noise reduction to enhance data quality.

### Quantitative Analysis
- **Metrics:** Detailed evaluation based on MSE, reconstruction accuracy, and curvature consistency.
- **Comparative Evaluation:** My approach demonstrates superior segmentation accuracy compared to traditional 3D CNN methods.

### Qualitative Outcomes
- **Visualizations:** High-resolution renderings using polyscope and matplotlib underscore the robustness of the segmentation.
- **Multi-view Integration:** Consensus across different perspectives leads to a comprehensive and accurate 3D reconstruction.

### Final Results
After training and extensive evaluation, the final outputs have been stored in the results directory. Here, you can find:
- High-resolution multi-view projection maps (combined_maps.png).
- The final segmentation model (parcellation_model.pth) with advanced legends for regional annotations.
- Quantitative evaluation metrics, including Mean Squared Error (MSE) and segmentation accuracy, demonstrating the robustness and efficiency of the proposed pipeline.
These results confirm that the 2D projection-based approach yields a precisely segmented brain.
![Final Resuts](<results/Final Results.png>)

## Discussion
My proposed methodology offers several significant advantages:
- **Efficiency:** Operating in the 2D projection space reduces computational complexity.
- **Robustness:** Multi-view integration enhances segmentation accuracy and consistency.
- **Scalability:** The framework can be extended to incorporate multi-modal imaging data and larger datasets.

However, challenges remain in handling extreme anatomical variations and ensuring projection quality across diverse datasets.

## Conclusion
I present a comprehensive 2D projection-based framework for cortical mesh segmentation that outperforms conventional approaches in efficiency and accuracy. This work lays the foundation for future advancements in neuroimaging segmentation, including integration with additional imaging modalities and real-time clinical applications.

## Future Work
- Integrate multi-modal imaging data for enriched segmentation.
- Enhance network architectures using attention mechanisms.
- Develop interactive visualization tools for clinical and research applications.

## Acknowledgements
I gratefully acknowledge the support of [Dr. Karthik Gopinath](https://lcn.martinos.org/people/karthik-gopinath/) at Harvard Medical School and Massachussets General Hospital. Special thanks to the open-source community for [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) and the scientific libraries that powered this project.

## Contact
For further inquiries, please contact me at [sergiusnyah@gmail.com](mailto:sergiusnyah@gmail.com)

Complete Article: [Pseudo Rendering: A Novel Deep Learning Approach for 2D Cortical Mesh Segmentation (MIT SGI 2024)](https://summergeometry.org/sgi2024/pseudo-rendering-a-novel-deep-learning-approach-for-2d-cortical-mesh-segmentation/)
