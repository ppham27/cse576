#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <QtGui>

#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"

/***********************************************************************
  This is the only file you need to change for your assignment. The
  other files control the UI (in case you want to make changes.)
************************************************************************/

/***********************************************************************
  The first eight functions provide example code to help get you started
************************************************************************/


// Convert an image to grayscale
void MainWindow::BlackWhiteImage(QImage *image)
{
    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int noiseMag = mag*2;

    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            int red = qRed(pixel), green = qGreen(pixel), blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            // otherwise add the same amount of noise to each channel
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;
                red += noise; green += noise; blue += noise;
            }
            image->setPixel(c, r, qRgb(max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))));
        }
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    int w = image.width();
    int h = image.height();
    QImage buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(int r=0;r<h/2;r++)
        for(int c=0;c<w/2;c++)
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
}

// Round float values to the nearest integer values and make sure the value lies in the range [0,255]
QRgb restrictColor(double red, double green, double blue)
{
    int r = (int)(floor(red+0.5));
    int g = (int)(floor(green+0.5));
    int b = (int)(floor(blue+0.5));

    return qRgb(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)));
}

// Normalize the values of the kernel to sum-to-one
void NormalizeKernel(double *kernel, int kernelWidth, int kernelHeight)
{
    double denom = 0.000001; int i;
    for(i=0; i<kernelWidth*kernelHeight; i++)
        denom += kernel[i];
    for(i=0; i<kernelWidth*kernelHeight; i++)
        kernel[i] /= denom;
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it's not done here.
// Note: This function is written using QImage form of the input image. But all other functions later use the double form
void MainWindow::MeanBlurImage(QImage *image, int radius)
{
    if(radius == 0)
        return;
    int size = 2*radius + 1; // This is the size of the kernel

    // Note: You can access the width and height using 'imageWidth' and 'imageHeight' respectively in the functions you write
    int w = image->width();
    int h = image->height();

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders (zero-padding)
    QImage buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute the kernel to convolve with the image
    double *kernel = new double [size*size];

    for(int i=0;i<size*size;i++)
        kernel[i] = 1.0;

    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // For each pixel in the image...
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            double rgb[3];
            rgb[0] = rgb[1] = rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(int rd=-radius;rd<=radius;rd++)
                for(int cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     //For the functions you write, check the ConvertQImage2Double function to see how to get the pixel value
                     QRgb pixel = buffer.pixel(c + cd + radius, r + rd + radius);

                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }
            // Store the pixel in the image to be returned
            // You need to store the RGB values in the double form of the image
            image->setPixel(c, r, restrictColor(rgb[0],rgb[1],rgb[2]));
        }
    }
    // Clean up (use this carefully)
    delete[] kernel;
}

// Convert QImage to a matrix of size (imageWidth*imageHeight)*3 having double values
void MainWindow::ConvertQImage2Double(QImage image)
{
    // Global variables to access image width and height
    imageWidth = image.width();
    imageHeight = image.height();

    // Initialize the global matrix holding the image
    // This is how you will be creating a copy of the original image inside a function
    // Note: 'Image' is of type 'double**' and is declared in the header file (hence global variable)
    // So, when you create a copy (say buffer), write "double** buffer = new double ....."
    Image = new double* [imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
            Image[i] = new double[3];

    // For each pixel
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
        {
            // Get a pixel from the QImage form of the image
            QRgb pixel = image.pixel(c,r);

            // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
            Image[r*imageWidth+c][0] = (double) qRed(pixel);
            Image[r*imageWidth+c][1] = (double) qGreen(pixel);
            Image[r*imageWidth+c][2] = (double) qBlue(pixel);
        }
}

// Convert the matrix form of the image back to QImage for display
void MainWindow::ConvertDouble2QImage(QImage *image)
{
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
            image->setPixel(c, r, restrictColor(Image[r*imageWidth+c][0], Image[r*imageWidth+c][1], Image[r*imageWidth+c][2]));

    for (int i = 0; i < imageWidth*imageHeight; ++i) delete[] Image[i];
    delete[] Image;
}


/**************************************************
 TIME TO WRITE CODE
**************************************************/

/**************************************************
 TASK 1
**************************************************/
namespace {
  enum PaddingScheme {
    kZero = 0,
    kReflected = 1,
  };

  void MakePaddedBuffer(const double* const* image, int imageWidth, int imageHeight,
                        int kernelWidth, int kernelHeight, PaddingScheme paddingScheme,
                        double*** imageBuffer, int* imageBufferWidth, int* imageBufferHeight) {
    // Buffer image. If kernel is even-sized add extra pixel to top and left.
    int padding[4] = {/*top=*/kernelHeight/2,
                      /*right=*/(kernelWidth - 1)/2,
                      /*bottom=*/(kernelHeight - 1)/2,
                      /*left=*/kernelWidth/2};
    *imageBufferWidth = padding[1] + imageWidth + padding[3];
    *imageBufferHeight = padding[0] + imageHeight + padding[2];
    *imageBuffer = new double*[(*imageBufferWidth) * (*imageBufferHeight)];
    for (int i = 0; i < *imageBufferWidth; ++i) {
      for (int j = 0; j < *imageBufferHeight; ++j) {
        if (i < padding[3] || i >= imageWidth + padding[3] ||
            j < padding[0] || j >= imageHeight + padding[0]) {
          (*imageBuffer)[j * (*imageBufferWidth) + i] = new double[3]{0, 0, 0};
          // Alternative padding schemes.
          if (paddingScheme == PaddingScheme::kReflected) {
            int k = i < padding[3] ? padding[3] - i :  // Left.
              i >= imageWidth + padding[3] ? 2*imageWidth + padding[3] - i - 1 :  // Right.
              i - padding[3];  // In range.
            k = std::min(std::max(0, k), imageWidth - 1);
            int l = j < padding[0] ? padding[0] - j :  // Above.
              j >= imageHeight + padding[0] ? 2*imageHeight + padding[0] - j - 1 :  // Below.
              j - padding[0];  // In range.
            l = std::min(std::max(0, l), imageHeight - 1);
            std::memcpy((*imageBuffer)[j * (*imageBufferWidth) + i], image[l*imageWidth + k],
                        3*sizeof(double));
          }
        } else {
          (*imageBuffer)[j * (*imageBufferWidth) + i] = new double[3];
          std::memcpy((*imageBuffer)[j * (*imageBufferWidth) + i],
                      image[(j - padding[0])*imageWidth + (i - padding[3])],
                      3*sizeof(double));
        }
      }
    }    
  }
}  // namespace

// Convolve the image with the kernel
void MainWindow::Convolution(double** image, double *kernel, int kernelWidth, int kernelHeight, bool add)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * kernel: 1-D array of kernel values
 * kernelWidth: width of the kernel
 * kernelHeight: height of the kernel
 * add: a boolean variable (taking values true or false)
*/
{
  Q_ASSERT_X(kernelWidth > 0, "convolution", "kernel width must be positive.");
  Q_ASSERT_X(kernelHeight > 0, "convolution", "kernel height must be positive.");
  // Allocate buffer with zero padding. Reflected padding can specified with paddingScheme.
  double **imageBuffer;
  int imageBufferWidth, imageBufferHeight;
  ::MakePaddedBuffer(image, imageWidth, imageHeight, kernelWidth, kernelHeight,
                     PaddingScheme::kZero,  // Change to kReflected for reflected padding.
                     &imageBuffer, &imageBufferWidth, &imageBufferHeight);
  // Convolve image.
  for (int i = 0; i < imageWidth; ++i) {
    for (int j = 0; j < imageHeight; ++j) {
      for (int c = 0; c < 3; ++c) {
        image[j*imageWidth + i][c] = add ? 128 : 0;
        for (int k = 0; k < kernelWidth; ++k) {
          int x = i + k;
          for (int l = 0; l < kernelHeight; ++l) {
            int y = j + l;
            image[j*imageWidth + i][c] +=
              kernel[l*kernelWidth + k]*imageBuffer[y*imageBufferWidth + x][c];
          }
        }
      }
    }
  }
  // Clean up.
  for (int i = 0; i < imageBufferWidth * imageBufferHeight; ++i) delete[] imageBuffer[i];
  delete[] imageBuffer;
}

/**************************************************
 TASK 2
**************************************************/

// Apply the 2-D Gaussian kernel on an image to blur it
void MainWindow::GaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
  int radius = static_cast<int>(3 * ceil(sigma));
  int size = 2 * radius + 1;
  double sigma2 = sigma*sigma;
  double* kernel = new double[size*size];
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int r = i - radius, c = j - radius;
      kernel[j*size + i] = exp(-(r*r + c*c)/(2*sigma2))/(2*M_PI*sigma2);
    }
  }
  NormalizeKernel(kernel, size, size);
  Convolution(image, kernel, /*kernelWidth=*/size, /*kernelHeight=*/size, /*add=*/false);
  delete[] kernel;
}

/**************************************************
 TASK 3
**************************************************/

// Perform the Gaussian Blur first in the horizontal direction and then in the vertical direction
void MainWindow::SeparableGaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
  int radius = static_cast<int>(3 * ceil(sigma));
  int size = 2 * radius + 1;
  double sigma2 = sigma*sigma;
  double* kernel = new double[size];
  for (int i = 0; i < size; ++i) {
    int x = i - radius;
    kernel[i] = exp(-(x*x)/(2*sigma2))/(sigma*sqrt(2*M_PI));
  }
  NormalizeKernel(kernel, size, 1);
  Convolution(image, kernel, size, 1, false);
  Convolution(image, kernel, 1, size, false);
  delete[] kernel;
}

/********** TASK 4 (a) **********/

// Compute the First derivative of an image along the horizontal direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_x(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
  double *kernel = new double[3]{-1, 0, 1};
  Convolution(image, kernel, 3, 1, true);
  GaussianBlurImage(image, sigma);
  delete[] kernel;
}

/********** TASK 4 (b) **********/

// Compute the First derivative of an image along the vertical direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_y(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
  double *kernel = new double[3]{-1, 0, 1};
  Convolution(image, kernel, 1, 3, true);
  GaussianBlurImage(image, sigma);
  delete[] kernel;
}

/********** TASK 4 (c) **********/

// Compute the Second derivative of an image using the Laplacian operator and then apply Gaussian blur
void MainWindow::SecondDerivImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
  double *kernel = new double[9]{0, 1, 0, 1, -4, 1, 0, 1, 0};
  Convolution(image, kernel, 3, 3, true);
  GaussianBlurImage(image, sigma);
  delete[] kernel;
}

/**************************************************
 TASK 5
**************************************************/

// Sharpen an image by subtracting the image's second derivative from the original image
void MainWindow::SharpenImage(double** image, double sigma, double alpha)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
 * alpha: constant by which the second derivative image is to be multiplied to before subtracting it from the original image
*/
{
  double** imageSecondDeriv = new double*[imageWidth*imageHeight];
  for (int i = 0; i < imageWidth*imageHeight; ++i) {
    imageSecondDeriv[i] = new double[3];
    std::memcpy(imageSecondDeriv[i], image[i], 3*sizeof(double));
  }
  SecondDerivImage(imageSecondDeriv, sigma);
  for (int i = 0; i < imageWidth*imageHeight; ++i)
    for (int c = 0; c < 3; ++c) image[i][c] -= alpha*(imageSecondDeriv[i][c] - 128);

  for (int i = 0; i < imageWidth*imageHeight; ++i) delete[] imageSecondDeriv[i];
  delete[] imageSecondDeriv;
}

/**************************************************
 TASK 6
**************************************************/

// Display the magnitude and orientation of the edges in an image using the Sobel operator in both X and Y directions
void MainWindow::SobelImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
*/
{
  double** imageX = new double*[imageWidth*imageHeight];
  double** imageY = new double*[imageWidth*imageHeight];
  for (int i = 0; i < imageWidth*imageHeight; ++i) {
    imageX[i] = new double[3];
    imageY[i] = new double[3];
    std::memcpy(imageX[i], image[i], 3*sizeof(double));
    std::memcpy(imageY[i], image[i], 3*sizeof(double));
  }
  double* kernelX = new double[9]{-1,0,1,-2,0,2,-1,0,1};
  double* kernelY = new double[9]{1,2,1,0,0,0,-1,-2,-1};

  Convolution(imageX, kernelX, 3, 3, false);
  Convolution(imageY, kernelY, 3, 3, false);
  for (int i = 0; i < imageWidth*imageHeight; ++i) {
    // Divide magnitude by 8 to avoid spurious edges.
    double magnitude = sqrt(imageX[i][0]*imageX[i][0] + imageY[i][0]*imageY[i][0])/8;
    double orientation = atan2(imageY[i][0], imageX[i][0]);
    // The following 3 lines of code to set the image pixel values after
    // computing magnitude and orientation (sin(orien) + 1)/2 converts the sine
    // value to the range [0,1]. Similarly for cosine.
    image[i][0] = magnitude*4.0*(sin(orientation) + 1.0)/2.0;
    image[i][1] = magnitude*4.0*(cos(orientation) + 1.0)/2.0;
    image[i][2] = magnitude*4.0 - image[i][0] - image[i][1];
  }

  delete[] kernelX;
  delete[] kernelY;
  for (int i = 0; i < imageWidth*imageHeight; ++i) {
    delete[] imageX[i]; delete[] imageY[i];
  }
  delete[] imageX;
  delete[] imageY;
}

/**************************************************
 TASK 7
**************************************************/

// Compute the RGB values at a given point in an image using bilinear interpolation.
void MainWindow::BilinearInterpolation(double** image, double x, double y, double rgb[3])
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * x: x-coordinate (corresponding to columns) of the position whose RGB values are to be found
 * y: y-coordinate (corresponding to rows) of the position whose RGB values are to be found
 * rgb[3]: array where the computed RGB values are to be stored
*/
{
  int x1 = static_cast<int>(floor(x)), x2 = static_cast<int>(ceil(x));
  int y1 = static_cast<int>(floor(y)), y2 = static_cast<int>(ceil(y));
  // Out of bounds.
  if (x1 < 0 || y1 < 0 || x2 >= imageWidth || y2 >= imageHeight) {
    rgb[0] = rgb[1] = rgb[2] = 0;
    return;
  }
  // Account for various cases where denominator might be 0.
  if (x1 == x2 && y1 == y2) {
    std::memcpy(rgb, image[imageWidth*y1 + x1], 3*sizeof(double));
    return;
  }
  if (y1 == y2) {
    for (int c = 0; c < 3; ++c)
      rgb[c] = (image[imageWidth*y1 + x1][c]*(x2 - x) + image[imageWidth*y1 + x2][c]*(x-x1))/(x2-x1);
    return;
  }
  if (x1 == x2) {
    for (int c = 0; c < 3; ++c)
      rgb[c] = (image[imageWidth*y1 + x1][c]*(y2 - y) + image[imageWidth*y2 + x1][c]*(y-y1))/(y2-y1);
    return;
  }
  // General case.
  for (int c = 0; c < 3; ++c) {
    rgb[c] = (image[imageWidth*y1 + x1][c]*(x2-x)*(y2-y) +
              image[imageWidth*y1 + x2][c]*(x-x1)*(y2-y) +
              image[imageWidth*y2 + x1][c]*(x2-x)*(y-y1) +
              image[imageWidth*y2 + x2][c]*(x-x1)*(y-y1))/(x2-x1)/(y2-y1);
  }
}

/*******************************************************************************
 Here is the code provided for rotating an image. 'orien' is in degrees.
********************************************************************************/

// Rotating an image by "orien" degrees
void MainWindow::RotateImage(double** image, double orien) {
  double radians = -2.0*3.141*orien/360.0;

  // Make a copy of the original image and then re-initialize the original image with 0
  double** buffer = new double* [imageWidth*imageHeight];
  for (int i = 0; i < imageWidth*imageHeight; i++) {
    buffer[i] = new double [3];
    std::memcpy(buffer[i], image[i], 3*sizeof(double));
    std::fill_n(image[i], 3, 0);  // re-initialize to 0.
  }

  for (int r = 0; r < imageHeight; r++)
    for (int c = 0; c < imageWidth; c++) {
      // Rotate around the center of the image
      double x0 = (double) (c - imageWidth/2);
      double y0 = (double) (r - imageHeight/2);
      // Rotate using rotation matrix
      double x1 = x0*cos(radians) - y0*sin(radians);
      double y1 = x0*sin(radians) + y0*cos(radians);
      x1 += (double) (imageWidth/2);
      y1 += (double) (imageHeight/2);

      double rgb[3];
      BilinearInterpolation(buffer, x1, y1, rgb);

      // Note: "image[r*imageWidth+c] = rgb" merely copies the head pointers of the arrays, not the values
      image[r*imageWidth+c][0] = rgb[0];
      image[r*imageWidth+c][1] = rgb[1];
      image[r*imageWidth+c][2] = rgb[2];
    }
    
  for (int i = 0; i < imageWidth*imageHeight; ++i) delete[] buffer[i];
  delete[] buffer;
}

/**************************************************
 TASK 8
**************************************************/

// Find the peaks of the edge responses perpendicular to the edges
void MainWindow::FindPeaksImage(double** image, double thres)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
 * thres: threshold value for magnitude
*/
{
  // First apply Sobel edge detector to get magnitudes and orientations.
  double** sobel = new double*[imageWidth*imageHeight];
  for (int i = 0; i < imageWidth*imageHeight; ++i) {
    sobel[i] = new double[3];
    std::memcpy(sobel[i], image[i], 3*sizeof(double));
  }
  SobelImage(sobel);
  for (int i = 0; i < imageWidth*imageHeight; ++i) {
    // Reverse engineer Sobel calculation to restore magnitude.
    sobel[i][2] += sobel[i][0] + sobel[i][1]; sobel[i][2] *= 2;
  }
  // Using magnitudes and peaks to find orientation.
  for (int i = 0; i < imageWidth*imageHeight; ++i) {
    // Zero out pixels that aren't peaks and pixels with edge magnitudes less than threshold.
    image[i][0] = image[i][1] = image[i][2] = 0;
    if (sobel[i][2] <= thres) continue;
    // Reverse engineer Sobel calculation to obtain sine and cosine of orientation.
    double sin_theta = 4*sobel[i][0]/sobel[i][2] - 1;
    double cos_theta = 4*sobel[i][1]/sobel[i][2] - 1;
    // Calculate edge magnitudes with interpolation.
    int r = i / imageWidth, c = i % imageWidth;
    double rgb[3];
    BilinearInterpolation(sobel, c + cos_theta, r + sin_theta, rgb);
    double magnitude_e1 = rgb[2];
    BilinearInterpolation(sobel, c - cos_theta, r - sin_theta, rgb);
    double magnitude_e2 = rgb[2];
    // Whiten peak respone pixels.
    if (sobel[i][2] >= magnitude_e1 && sobel[i][2] >= magnitude_e2)
      image[i][0] = image[i][1] = image[i][2] = 255;
  }
  for (int i = 0; i < imageWidth*imageHeight; ++i) delete[] sobel[i];
  delete[] sobel;
}

/**************************************************
 TASK 9 (a)
**************************************************/

namespace {
  void KMeans(double* const* clusters, int num_clusters,
              double* const* image, int imageWidth, int imageHeight) {
    int* cluster_assignments = new int[imageWidth*imageHeight];
    int iteration = 0, max_iterations = 100;
    long long total_distance_delta = std::numeric_limits<long long>::max(),
      previous_total_distance = std::numeric_limits<long long>::max(), epsilon = 30;
    while (++iteration <= max_iterations && total_distance_delta >= epsilon*num_clusters) {
      // Accumulate new clusters.
      std::vector<std::array<double, 3>> new_clusters(num_clusters);
      for (int k = 0; k < num_clusters; ++k) new_clusters[k].fill(0);
      std::vector<int> cluster_count(num_clusters, 0);
      // Assign clusters and Keep track of sum of pixel minimum distance to a cluster center.
      long long total_distance = 0;
      for (int i = 0; i < imageWidth*imageHeight; ++i) {
        int min_distance = std::numeric_limits<int>::max(), cluster = -1;
        for (int k = 0; k < num_clusters; ++k) {
          int distance = 0;
          for (int c = 0; c < 3; ++c) distance += std::abs(image[i][c] - clusters[k][c]);
          if (distance < min_distance) min_distance = distance, cluster = k;
        }
        Q_ASSERT_X(cluster > -1, "kmeans", "No closest cluster was found.");
        cluster_assignments[i] = cluster;
        ++cluster_count[cluster];
        for (int c = 0; c < 3; ++c) new_clusters[cluster][c] += image[i][c];
        total_distance += min_distance;
      }
      // Update clusters with new means.
      for (int k = 0; k < num_clusters; ++k)
        if (cluster_count[k] > 0)
          for (int c = 0; c < 3; ++c) clusters[k][c] = new_clusters[k][c]/cluster_count[k];
      total_distance_delta = std::abs(previous_total_distance - total_distance);
      previous_total_distance = total_distance;
    }

    for (int i = 0; i < imageWidth*imageHeight; ++i)
      std::memcpy(image[i], clusters[cluster_assignments[i]], 3*sizeof(double));

    delete[] cluster_assignments;
  }
}  // namespace

// Perform K-means clustering on a color image using random seeds
void MainWindow::RandomSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{  
  std::mt19937_64 generator((std::random_device())());
  std::uniform_int_distribution<double> pixel_dist(0, 255);

  double** clusters = new double*[num_clusters];
  for (int k = 0; k < num_clusters; ++k) {
    clusters[k] = new double[3]{
      pixel_dist(generator), pixel_dist(generator), pixel_dist(generator)};
  }

  KMeans(clusters, num_clusters, image, imageWidth, imageHeight);

  for (int k = 0; k < num_clusters; ++k) delete[] clusters[k];
  delete[] clusters;
}

/**************************************************
 TASK 9 (b)
**************************************************/

// Perform K-means clustering on a color image using seeds from the image itself
void MainWindow::PixelSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
  std::vector<int> pixel_indices; pixel_indices.reserve(imageWidth*imageHeight);
  for (int i = 0; i < imageWidth*imageHeight; ++i) pixel_indices.push_back(i);

  std::mt19937_64 generator((std::random_device())());
  std::shuffle(pixel_indices.begin(), pixel_indices.end(), generator);

  num_clusters = std::min(num_clusters, static_cast<int>(pixel_indices.size()));
  double** clusters = new double*[num_clusters];
  for (int k = 0; k < num_clusters; ++k) {
    clusters[k] = new double[3];
    std::memcpy(clusters[k], image[pixel_indices[k]], 3*sizeof(double));
  }

  KMeans(clusters, num_clusters, image, imageWidth, imageHeight);

  for (int k = 0; k < num_clusters; ++k) delete[] clusters[k];
  delete[] clusters;
}


/**************************************************
 EXTRA CREDIT TASKS
**************************************************/

// Perform K-means clustering on a color image using the color histogram
void MainWindow::HistogramSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
}

// Apply the median filter on a noisy image to remove the noise
void MainWindow::MedianImage(double** image, int radius)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * radius: radius of the kernel
*/
{
    // Add your code here
}

// Apply Bilater filter on an image
void MainWindow::BilateralImage(double** image, double sigmaS, double sigmaI)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigmaS: standard deviation in the spatial domain
 * sigmaI: standard deviation in the intensity/range domain
*/
{
    // Add your code here.  Should be similar to GaussianBlurImage.
}

// Perform the Hough transform
void MainWindow::HoughImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}

// Perform smart K-means clustering
void MainWindow::SmartKMeans(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}
