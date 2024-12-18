# Singular Value Decomposition (SVD) for Image Compression

## Description

This project demonstrates the application of Singular Value Decomposition (SVD) on an image for dimensionality reduction and image compression. SVD is a powerful linear algebra tool used for matrix factorization, which can be employed for tasks like noise reduction, feature extraction, and image compression. In this case, we apply SVD to an image, compress it by retaining only a few singular values, and compare the quality of the compressed image to the original one.

### Key Steps:
1. **Image Download**: The image is downloaded from a URL using the `requests` library.
2. **Image Processing**: The downloaded image is read and converted into grayscale using OpenCV.
3. **Singular Value Decomposition (SVD)**: The image matrix is decomposed into three matrices—U, S, and V—using NumPy's `linalg.svd()` function.
4. **Variance Explained**: The variance explained by each singular value is computed and visualized in a bar plot.
5. **Image Reconstruction**: The image is reconstructed using a low-rank approximation by selecting a subset of singular values. The quality of the image is compared by showing reconstructed images with varying numbers of components.

## Algorithm: Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) decomposes a matrix \( A \) into three matrices:

\[
A = U \cdot S \cdot V^T
\]

Where:
- \( U \) is an orthogonal matrix containing the left singular vectors.
- \( S \) is a diagonal matrix containing the singular values, ordered from largest to smallest.
- \( V^T \) is the transpose of the orthogonal matrix containing the right singular vectors.

### Image Compression Using SVD
In the context of image processing, SVD allows for dimensionality reduction, where we can retain only the most significant singular values, thereby compressing the image. The number of singular values used affects the quality of the reconstructed image: fewer singular values lead to higher compression but more loss of detail.

#### Variance Explained
The variance explained by each singular value indicates how much information or detail is retained by each singular vector. The first singular values contribute most of the image's information, while the later ones represent finer details or noise.

### Image Reconstruction
Reconstructing the image with fewer singular values results in a compressed, low-rank approximation. This approximation can be used to assess how much detail is lost when reducing the number of components.

## Code Explanation

1. **Image Download**:
   - The image is downloaded using the `requests` library.
   - It is saved locally as `image.png`.

2. **Image Processing**:
   - The image is read using OpenCV and converted to grayscale for simplicity in SVD computation.

3. **SVD Decomposition**:
   - The image matrix is decomposed using `np.linalg.svd()`, which returns the matrices \( U \), \( S \), and \( V \).
   - The shapes of these matrices are printed for insight into the decomposition.

4. **Variance Explained**:
   - The variance explained by each singular value is calculated and plotted for the first 20 singular values.

5. **Low-Rank Approximation**:
   - The image is reconstructed using a different number of components (e.g., 1, 5, 10, 15, 20, etc.), and the results are shown in subplots.

## Results: Variance Explained

The variance explained by the first 20 singular values is calculated and displayed in a bar chart. This chart allows us to see how much information each singular value captures. The top singular values typically explain the most variance.

## Image Compression Example

The reconstructed images with different numbers of components (from 1 to 20) demonstrate the effect of reducing the rank on the image quality. As the number of components decreases, the image becomes increasingly blurry, indicating the loss of detail.

## Example Image:

Below is an example of how SVD works visually. The first image is the original, and the subsequent images show the approximation using fewer components.

![SVD Compression](https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Singular_Value_Decomposition.svg/600px-Singular_Value_Decomposition.svg.png)

## Comparison with Other Models

- **Principal Component Analysis (PCA)**:
  - PCA is often compared with SVD, as both involve dimensionality reduction and are based on eigenvalue decomposition. However, PCA focuses on maximizing variance, while SVD decomposes the matrix in a more generalized way.
  - In image compression, PCA and SVD both yield similar results, but SVD is more computationally efficient for large matrices, especially when working with sparse data.

- **Other Compression Algorithms**:
  - **JPEG Compression**: Unlike SVD-based compression, JPEG uses a discrete cosine transform (DCT) and quantization for compression, which is optimized for lossy compression. SVD may provide higher-quality results in certain applications, as it uses matrix decomposition for capturing global patterns.
  - **Wavelet Compression**: Wavelet-based methods use multi-resolution decomposition to capture features at various scales. While wavelets are better at preserving details in certain scenarios, SVD offers a more direct method for dimensionality reduction in image processing.

## Conclusion

This project illustrates how Singular Value Decomposition (SVD) can be applied to image compression. By reducing the number of singular values, we achieve varying levels of image quality and compression. SVD offers an effective approach to dimensionality reduction, and the results show that fewer components lead to higher compression at the cost of detail loss. SVD-based image compression can be compared with other methods like PCA and JPEG, offering a good balance between compression efficiency and image quality.

## Requirements
- `opencv-python`
- `numpy`
- `requests`
- `matplotlib`
- `seaborn`

Install the required libraries using:

```bash
pip install opencv-python numpy requests matplotlib seaborn