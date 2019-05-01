#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <iterator>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "Matrix.h"

/*******************************************************************************
    The following are helper routines with code already written.
    The routines you'll need to write for the assignment are below.
*******************************************************************************/

/*******************************************************************************
Blur a single channel floating point image with a Gaussian.
    image - input and output image
    w - image width
    h - image height
    sigma - standard deviation of Gaussian
*******************************************************************************/
void MainWindow::GaussianBlurImage(double *image, int w, int h, double sigma)
{
    int r, c, rd, cd, i;
    int radius = max(1, (int) (sigma*3.0));
    int size = 2*radius + 1;
    double *buffer = new double [w*h];

    memcpy(buffer, image, w*h*sizeof(double));

    if(sigma == 0.0)
        return;

    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        double dist = (double) (i - radius);

        kernel[i] = exp(-(dist*dist)/(2.0*sigma*sigma));
    }

    double denom = 0.000001;

    for(i=0;i<size;i++)
        denom += kernel[i];
    for(i=0;i<size;i++)
        kernel[i] /= denom;

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double val = 0.0;
            double denom = 0.0;

            for(rd=-radius;rd<=radius;rd++)
                if(r + rd >= 0 && r + rd < h)
                {
                     double weight = kernel[rd + radius];

                     val += weight*buffer[(r + rd)*w + c];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }

    memcpy(buffer, image, w*h*sizeof(double));

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double val = 0.0;
            double denom = 0.0;

            for(cd=-radius;cd<=radius;cd++)
                if(c + cd >= 0 && c + cd < w)
                {
                     double weight = kernel[cd + radius];

                     val += weight*buffer[r*w + c + cd];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }


    delete [] kernel;
    delete [] buffer;
}


/*******************************************************************************
Bilinearly interpolate image (helper function for Stitch)
    image - input image
    (x, y) - location to interpolate
    rgb - returned color values
*******************************************************************************/
bool MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{

    int r = (int) y;
    int c = (int) x;
    double rdel = y - (double) r;
    double cdel = x - (double) c;
    QRgb pixel;
    double del;

    rgb[0] = rgb[1] = rgb[2] = 0.0;

    if(r >= 0 && r < image->height() - 1 && c >= 0 && c < image->width() - 1)
    {
        pixel = image->pixel(c, r);
        del = (1.0 - rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r);
        del = (1.0 - rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c, r+1);
        del = (rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r+1);
        del = (rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);
    }
    else
        return false;

    return true;
}


/*******************************************************************************
Draw detected Harris corners
    cornerPts - corner points
    numCornerPts - number of corner points
    imageDisplay - image used for drawing

    Draws a red cross on top of detected corners
*******************************************************************************/
void MainWindow::DrawCornerPoints(CIntPt *cornerPts, int numCornerPts, QImage &imageDisplay)
{
   int i;
   int r, c, rd, cd;
   int w = imageDisplay.width();
   int h = imageDisplay.height();

   for(i=0;i<numCornerPts;i++)
   {
       c = (int) cornerPts[i].m_X;
       r = (int) cornerPts[i].m_Y;

       for(rd=-2;rd<=2;rd++)
           if(r+rd >= 0 && r+rd < h && c >= 0 && c < w)
               imageDisplay.setPixel(c, r + rd, qRgb(255, 0, 0));

       for(cd=-2;cd<=2;cd++)
           if(r >= 0 && r < h && c + cd >= 0 && c + cd < w)
               imageDisplay.setPixel(c + cd, r, qRgb(255, 0, 0));
   }
}

namespace {
  inline double Intensity(QRgb pixel) {
    return qGreen(pixel);
  }

  constexpr int CUSTOM_DESC_SIZE = DESC_SIZE;
}

/*******************************************************************************
Compute corner point descriptors
    image - input image
    cornerPts - array of corner points
    numCornerPts - number of corner points

    If the descriptor cannot be computed, i.e. it's too close to the boundary of
    the image, its descriptor length will be set to 0.

    I've implemented a very simple 8 dimensional descriptor.  Feel free to
    improve upon this.
*******************************************************************************/
void MainWindow::ComputeDescriptors(QImage image, CIntPt *cornerPts, int numCornerPts) {
  int w = image.width();
  int h = image.height();
  double *buffer = new double[w*h];
  QRgb pixel;

  // Descriptor parameters
  double sigma = 2.0;

  // Computer descriptors from green channel
  for(int r = 0; r < h; r++)
    for(int c = 0; c < w; c++)
      buffer[r*w + c] = ::Intensity(image.pixel(c, r));

  // Blur
  GaussianBlurImage(buffer, w, h, sigma);

  // Compute the desciptor from the difference between the point sampled at its center
  // and eight points sampled around it. Go clockwise.
  int max_delta = 4;
#if DESC_SIZE == 16
  static const int rd_map[::CUSTOM_DESC_SIZE] = {-4, -4, -4, -4,  -4, -2,  0,  2, 4, 4,  4,  4,  4,  2,  0, -2};
  static const int cd_map[::CUSTOM_DESC_SIZE] = {-4, -2,  0,  2,   4,  4,  4,  4, 4, 2,  0, -2, -4, -4, -4, -4};
#elif DESC_SIZE == 8
  static const int rd_map[::CUSTOM_DESC_SIZE] = {-4, -4, -4,  0,  4,  4,  4,  0};
  static const int cd_map[::CUSTOM_DESC_SIZE] = {-4, -0,  4,  4,  4,  0, -4, -4};
#else
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
  static_assert(::CUSTOM_DESC_SIZE == 8 || ::CUSTOM_DESC_SIZE == 16,
                "Only descriptor sizes of 8 and 16 are supported, but DESC_SIZE is " STRINGIFY(DESC_SIZE) "."); 
  int rd_map[::CUSTOM_DESC_SIZE]; int cd_map[::CUSTOM_DESC_SIZE];
#endif
  for(int i = 0; i < numCornerPts; ++i) {
    int c = (int) cornerPts[i].m_X;
    int r = (int) cornerPts[i].m_Y;
    if(c >= max_delta && c < w - max_delta && r >= max_delta && r < h - max_delta) {
      double centerValue = buffer[(r)*w + c];
      for (int j = 0; j < ::CUSTOM_DESC_SIZE; ++j) {
        int rd = rd_map[j], cd = cd_map[j];
        cornerPts[i].m_Desc[j] = buffer[(r + rd)*w + c + cd] - centerValue;
      }
      cornerPts[i].m_DescSize = ::CUSTOM_DESC_SIZE;
    } else {
      cornerPts[i].m_DescSize = 0;
    }
  }

  delete [] buffer;
}

/*******************************************************************************
Draw matches between images
    matches - matching points
    numMatches - number of matching points
    image1Display - image to draw matches
    image2Display - image to draw matches

    Draws a green line between matches
*******************************************************************************/
void MainWindow::DrawMatches(CMatches *matches, int numMatches, QImage &image1Display, QImage &image2Display) {
    int i;
    // Show matches on image
    QPainter painter;
    painter.begin(&image1Display);
    QColor green(0, 250, 0);
    QColor red(250, 0, 0);

    for(i=0;i<numMatches;i++)
    {
        painter.setPen(green);
        painter.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter.setPen(red);
        painter.drawEllipse((int) matches[i].m_X1-1, (int) matches[i].m_Y1-1, 3, 3);
    }

    QPainter painter2;
    painter2.begin(&image2Display);
    painter2.setPen(green);

    for(i=0;i<numMatches;i++)
    {
        painter2.setPen(green);
        painter2.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter2.setPen(red);
        painter2.drawEllipse((int) matches[i].m_X2-1, (int) matches[i].m_Y2-1, 3, 3);
    }

}


/*******************************************************************************
Given a set of matches computes the "best fitting" homography
    matches - matching points
    numMatches - number of matching points
    h - returned homography
    isForward - direction of the projection (true = image1 -> image2, false = image2 -> image1)
*******************************************************************************/
bool MainWindow::ComputeHomography(CMatches *matches, int numMatches, double h[3][3], bool isForward) {
    int error;
    int nEq=numMatches*2;

    dmat M=newdmat(0,nEq,0,7,&error);
    dmat a=newdmat(0,7,0,0,&error);
    dmat b=newdmat(0,nEq,0,0,&error);

    double x0, y0, x1, y1;

    for (int i=0;i<nEq/2;i++)
    {
        if(isForward == false)
        {
            x0 = matches[i].m_X1;
            y0 = matches[i].m_Y1;
            x1 = matches[i].m_X2;
            y1 = matches[i].m_Y2;
        }
        else
        {
            x0 = matches[i].m_X2;
            y0 = matches[i].m_Y2;
            x1 = matches[i].m_X1;
            y1 = matches[i].m_Y1;
        }


        //Eq 1 for corrpoint
        M.el[i*2][0]=x1;
        M.el[i*2][1]=y1;
        M.el[i*2][2]=1;
        M.el[i*2][3]=0;
        M.el[i*2][4]=0;
        M.el[i*2][5]=0;
        M.el[i*2][6]=(x1*x0*-1);
        M.el[i*2][7]=(y1*x0*-1);

        b.el[i*2][0]=x0;
        //Eq 2 for corrpoint
        M.el[i*2+1][0]=0;
        M.el[i*2+1][1]=0;
        M.el[i*2+1][2]=0;
        M.el[i*2+1][3]=x1;
        M.el[i*2+1][4]=y1;
        M.el[i*2+1][5]=1;
        M.el[i*2+1][6]=(x1*y0*-1);
        M.el[i*2+1][7]=(y1*y0*-1);

        b.el[i*2+1][0]=y0;

    }
    int ret=solve_system (M,a,b);
    if (ret!=0)
    {
        freemat(M);
        freemat(a);
        freemat(b);

        return false;
    }
    else
    {
        h[0][0]= a.el[0][0];
        h[0][1]= a.el[1][0];
        h[0][2]= a.el[2][0];

        h[1][0]= a.el[3][0];
        h[1][1]= a.el[4][0];
        h[1][2]= a.el[5][0];

        h[2][0]= a.el[6][0];
        h[2][1]= a.el[7][0];
        h[2][2]= 1;
    }

    freemat(M);
    freemat(a);
    freemat(b);

    return true;
}


/*******************************************************************************
*******************************************************************************
*******************************************************************************

    The routines you need to implement are below

*******************************************************************************
*******************************************************************************
*******************************************************************************/

/*******************************************************************************
Detect Harris corners.
    image - input image
    sigma - standard deviation of Gaussian used to blur corner detector
    thres - Threshold for detecting corners
    cornerPts - returned corner points
    numCornerPts - number of corner points returned
    imageDisplay - image returned to display (for debugging)
*******************************************************************************/
void MainWindow::HarrisCornerDetector(QImage image, double sigma, double thres,
                                      CIntPt **cornerPts, int &numCornerPts, QImage &imageDisplay) {
    int r, c;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    // Compute the corner response using just the green channel
    for(r=0;r<h;r++)
      for(c=0;c<w;c++)
        buffer[r*w + c] = ::Intensity(image.pixel(c, r));

    // Blur before taking derivatives. Convolutions are commutative.
    GaussianBlurImage(buffer, w, h, sigma);
    // Gather up the derivatives in the general case.
    std::vector<std::vector<double>> x_derivative(h, std::vector<double>(c, 0));
    std::vector<std::vector<double>> y_derivative(h, std::vector<double>(c, 0));
    for (int r = 1; r < h - 1; ++r)
      for (int c = 1; c < w - 1; ++c) {
        x_derivative[r][c] = (buffer[r*w + c + 1] - buffer[r*w + c - 1])/2.0;
        y_derivative[r][c] = (buffer[(r+1)*w + c] - buffer[(r-1)*w + c])/2.0;
      }
    // Consider edge cases.
    if (h >= 2)
      for (int c = 0; c < w; ++c) {
        y_derivative[0][c] = buffer[w + c]/2.0;  // First row.
        y_derivative[h - 1][c] = -buffer[(h-2)*w + c]/2.0;  // Last row
      }    
    if (w > 2)
      for (int r = 0; r < h; ++r) {
        x_derivative[r][0] = buffer[r*w + 1]/2.0;  // First column.
        x_derivative[r][w - 1] = -buffer[r*w + (w-2)]/2.0;  // Last column.
      }
    // Compute the response function from the derivative.
    std::vector<std::vector<double>> response(h, std::vector<double>(c, 0));
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) {        
        double x_squared = x_derivative[r][c]*x_derivative[r][c];
        double y_squared = y_derivative[r][c]*y_derivative[r][c];
        double det = x_squared*y_squared - 2.*x_derivative[r][c]*y_derivative[r][c];
        double tr = x_squared + y_squared;
        if (tr != 0) response[r][c] = det/tr;
      }
    // Evaluate the responses for corner points.
    std::vector<std::pair<int, int>> corner_pts;
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        // Ignore points below threshold and trivial border corners.
        if (response[r][c] < thres ||
            (r == 0 && c == 0) || (r == 0 && c == w - 1) ||
            (r == h - 1 && c == 0) || (r == h - 1 && c == w - 1)) continue;
        // Do non-maximum suppression on a 5x5 window.
        int top = std::max(0, r - 2), right = std::min(w, c + 3),
          bottom = std::min(h, r + 3), left = std::max(0, c - 2);
        double max_response = 0;
        for (int i = top; i < bottom; ++i)
          for (int j = left; j < right; ++j) 
            if (i != r && j != c)
              max_response = std::max(response[i][j], max_response);
        if (response[r][c] > max_response) corner_pts.emplace_back(r, c);
      }
    }

    // Access the values using: (*cornerPts)[i].m_X = 5.0;
    //
    // The position of the corner point is (m_X, m_Y)
    // The descriptor of the corner point is stored in m_Desc
    // The length of the descriptor is m_DescSize, if m_DescSize = 0, then it is not valid.
    numCornerPts = corner_pts.size();
    *cornerPts = new CIntPt[numCornerPts];
    for (int i = 0; i < numCornerPts; ++i) {
      (*cornerPts)[i].m_X = static_cast<double>(corner_pts[i].second);
      (*cornerPts)[i].m_Y = static_cast<double>(corner_pts[i].first);
    }

    // Once you are done finding the corner points, display them on the image
    DrawCornerPoints(*cornerPts, numCornerPts, imageDisplay);

    delete [] buffer;
}

namespace {
  template<typename T, unsigned int D>
  class Index {
  public:
    explicit Index(std::vector<std::array<T, D>> points) :
      points_(points) {}

    Index() = delete;

    int nn_search(const T* query) const {
      std::pair<int, double> neighbor = std::make_pair(-1, std::numeric_limits<double>::max());
      for (int i = 0; i < points_.size(); ++i) {
        for (int k = 0; k < D; k += 2) {  // Rotates descriptor by k*2*PI/D.
          double distance = 0;
          for (int j = 0; j < D; ++j) distance += std::abs(query[j] - points_[i][(j + k) % D]);
          if (distance < neighbor.second) neighbor = std::make_pair(i, distance);
        }
      }
      return neighbor.first;
    }
    
  private:
    const std::vector<std::array<T, D>> points_;
  };
}

/*******************************************************************************
Find matching corner points between images.
    image1 - first input image
    cornerPts1 - corner points corresponding to image 1
    numCornerPts1 - number of corner points in image 1
    image2 - second input image
    cornerPts2 - corner points corresponding to image 2
    numCornerPts2 - number of corner points in image 2
    matches - set of matching points to be returned
    numMatches - number of matching points returned
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::MatchCornerPoints(QImage image1, CIntPt *cornerPts1, int numCornerPts1,
                                   QImage image2, CIntPt *cornerPts2, int numCornerPts2,
                                   CMatches **matches, int &numMatches,
                                   QImage &image1Display, QImage &image2Display) {
  // Compute the descriptors for each corner point.
  // You can access the descriptor for each corner point using cornerPts1[i].m_Desc[j].
  // If cornerPts1[i].m_DescSize = 0, it was not able to compute a descriptor for that point
  ComputeDescriptors(image1, cornerPts1, numCornerPts1);
  ComputeDescriptors(image2, cornerPts2, numCornerPts2);
  // Match points assuming one set of points subsets the other.
  CIntPt* &from_corner_pts = numCornerPts1 <= numCornerPts2 ? cornerPts1 : cornerPts2;
  CIntPt* &to_corner_pts = numCornerPts1 <= numCornerPts2 ? cornerPts2 : cornerPts1;
  std::function<int(CIntPt*&)> size = [&](CIntPt*& x) -> int {
    return x == cornerPts1 ? numCornerPts1 : numCornerPts2;
  };
  // Build an index to do a nearest neighbor search.
  std::vector<std::array<double, ::CUSTOM_DESC_SIZE>> points;
  for (int i = 0; i < size(to_corner_pts); ++i) {
    if (to_corner_pts[i].m_DescSize != ::CUSTOM_DESC_SIZE) continue;
    std::array<double, ::CUSTOM_DESC_SIZE> point;
    for (int j = 0; j < ::CUSTOM_DESC_SIZE; ++j) point[j] = to_corner_pts[i].m_Desc[j];
    points.push_back(std::move(point));
  }
  const ::Index<double, ::CUSTOM_DESC_SIZE> index(std::move(points));
  // Given two indices make a match.
  std::function<CMatches(int,int)> make_match =
    [&](int i, int j) -> CMatches {
    // The position of the corner point in image 1 is (m_X1, m_Y1)
    // The position of the corner point in image 2 is (m_X2, m_Y2)
    double x1 = from_corner_pts[i].m_X, y1 = from_corner_pts[i].m_Y;
    double x2 = to_corner_pts[j].m_X, y2 = to_corner_pts[j].m_Y;
    if (from_corner_pts == cornerPts2) { std::swap(x1, x2); std::swap(y1, y2); }
    return CMatches{x1, y1, x2, y2};
  };
  // Find all the matches with a nearest neighbor search.
  numMatches = 0;
  *matches = new CMatches[size(from_corner_pts)];  // Allocates potentially extra space.
  for (int i = 0; i < size(from_corner_pts); ++i) {
    if (!from_corner_pts[i].m_DescSize) continue;
    int neighbor = index.nn_search(from_corner_pts[i].m_Desc);
    (*matches)[numMatches++] = make_match(i, neighbor);
  }
  // Draw the matches
  DrawMatches(*matches, numMatches, image1Display, image2Display);
}

namespace {
  inline void Project(double x1, double y1, double &x2, double &y2, const double h[3][3]) {
    double w = x1*h[2][0] + y1*h[2][1] + h[2][2];
    x2 = (x1*h[0][0] + y1*h[0][1] + h[0][2])/w;
    y2 = (x1*h[1][0] + y1*h[1][1] + h[1][2])/w;
  }


  inline bool IsInlier(const CMatches& match, const double h[3][3], const double inlierThreshold) {
    double x3, y3;
    ::Project(match.m_X1, match.m_Y1, x3, y3, h);
    return std::abs(x3 - match.m_X2) + std::abs(y3 - match.m_Y2) <= inlierThreshold;
  }
}

/*******************************************************************************
Project a point (x1, y1) using the homography transformation h
    (x1, y1) - input point
    (x2, y2) - returned point
    h - input homography used to project point
*******************************************************************************/
void MainWindow::Project(double x1, double y1, double &x2, double &y2, double h[3][3]) {
  ::Project(x1, y1, x2, y2, h);  // Delegate to pure function.
}

/*******************************************************************************
Count the number of inliers given a homography.  This is a helper function for RANSAC.
    h - input homography used to project points (image1 -> image2
    matches - array of matching points
    numMatches - number of matchs in the array
    inlierThreshold - maximum distance between points that are considered to be inliers

    Returns the total number of inliers.
*******************************************************************************/
int MainWindow::ComputeInlierCount(double h[3][3], CMatches *matches, int numMatches, double inlierThreshold) {
  int num_inliers = 0;
  for (int i = 0; i < numMatches; ++i)
    if (::IsInlier(matches[i], h, inlierThreshold)) ++num_inliers;
  return num_inliers;
}

/*******************************************************************************
Compute homography transformation between images using RANSAC.
    matches - set of matching points between images
    numMatches - number of matching points
    numIterations - number of iterations to run RANSAC
    inlierThreshold - maximum distance between points that are considered to be inliers
    hom - returned homography transformation (image1 -> image2)
    homInv - returned inverse homography transformation (image2 -> image1)
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::RANSAC(CMatches *matches, int numMatches, int numIterations, double inlierThreshold,
                        double hom[3][3], double homInv[3][3], QImage &image1Display, QImage &image2Display) {
  std::mt19937_64 generator((std::random_device())());
  // Copy all the matches into inliers for shuffling.
  CMatches* inliers = new CMatches[numMatches];
  std::copy(matches, matches + numMatches, inliers);
  // Repeatedly shuffle and test homographies.
  int numInliers = 0;
  double testHom[3][3];
  for (int i = 0; i < numIterations; ++i) {
    std::shuffle(inliers, inliers + numMatches, generator);
    ComputeHomography(inliers, 4, testHom, true);
    int testNumInliers = ComputeInlierCount(testHom,  matches, numMatches, inlierThreshold);
    if (numInliers < testNumInliers) {
      numInliers = testNumInliers;
      for (int j = 0; j < 3; ++j) std::copy(testHom[j], testHom[j] + 3, hom[j]);
    }
  }
  // Gather up the inliers associated with the homography.
  int inlierIndex = 0;
  for (int i = 0; i < numMatches; ++i)
    if (::IsInlier(matches[i], hom, inlierThreshold)) inliers[inlierIndex++] = matches[i];
  Q_ASSERT_X(inlierIndex == numInliers, "RANSAC", "Inlier counts do not match.");
  // Compute a new homography.
  ComputeHomography(inliers, numInliers, hom, true);
  ComputeHomography(inliers, numInliers, homInv, false);
  // After you're done computing the inliers, display the corresponding matches.
  DrawMatches(inliers, numInliers, image1Display, image2Display);
}

namespace {
  double CenterWeight(double x, double y, double width, double height, unsigned int p) {    
    if (x < 0  || width <= x) qInfo() << x << ' ' << width << ' ' << height;
    Q_ASSERT_X(0 <= x && x < width, "CenterWeight", "x is not in range.");
    Q_ASSERT_X(0 <= y && y < height, "CenterWeight", "y is not in range.");
    double x_delta = std::min(x + 1., width - x);
    double y_delta = std::min(y + 1., height - y);
    return std::pow(std::min(x_delta, y_delta), p);
  }
}

/*******************************************************************************
Stitch together two images using the homography transformation
    image1 - first input image
    image2 - second input image
    hom - homography transformation (image1 -> image2)
    homInv - inverse homography transformation (image2 -> image1)
    stitchedImage - returned stitched image
*******************************************************************************/
void MainWindow::Stitch(QImage image1, QImage image2, double hom[3][3], double homInv[3][3], QImage &stitchedImage) {
    double xtl, ytl;  // Top left.
    ::Project(0, 0, xtl, ytl, homInv);

    double xtr, ytr;  // Top right.
    ::Project(image2.width() - 1, 0, xtr, ytr, homInv);

    double xbr, ybr;  // Bottom right.
    ::Project(image2.width() - 1, image2.height() - 1, xbr, ybr, homInv);

    double xbl, ybl;  // Bottom left.
    ::Project(0, image2.height() - 1, xbl, ybl, homInv);

    // Assume orientation is pretty much correct, so corners are in the same
    // relative position. In theory, a homography may flip or rotate.
    double projected_image2_scale = [=]() -> double {
      double projected_image2_width = std::max(xtr, xbr) - std::min(xtl, xbl);
      double projected_image2_height = std::max(ybl, ybr) - std::min(ytl, ytr);
      return std::sqrt((projected_image2_width/image2.width())*
                       (projected_image2_height/image2.height()));
    }();

    // Origin, width and height of stitchedImage.
    int xstl = std::min(std::min(0, static_cast<int>(floor(xtl))),
                        static_cast<int>(floor(xbl)));
    int ystl = std::min(std::min(0, static_cast<int>(floor(ytl))),
                        static_cast<int>(floor(ytr)));
    int xsbr = std::max(std::max(image1.width() - 1, static_cast<int>(floor(xtr))),
                        static_cast<int>(floor(xbr))) + 1;
    int ysbr = std::max(std::max(image1.height() - 1, static_cast<int>(floor(ybl))),
                        static_cast<int>(floor(ybr))) + 1;
    int ws = xsbr - xstl;
    int hs = ysbr - ystl;
    stitchedImage = QImage(ws, hs, QImage::Format_RGB32);
    stitchedImage.fill(qRgb(0,0,0));
    // Translate image1 into the stiched image.
    for (int j = 0; j < image1.width(); ++j)
      for (int i = 0; i < image1.height(); ++i)
        stitchedImage.setPixel(j - xstl, i - ystl, image1.pixel(j, i));
    // Project image2 onto the stiched image.
    for (int j = 0; j < ws; ++j)
      for (int i = 0; i < hs; ++i) {
        double rgb[3];
        double x, y;
        int image1_x = j + xstl, image1_y = i + ystl;  // Translate back to image1 before projecting.
        ::Project(image1_x, image1_y, x, y, hom);
        QRgb pixel1 = stitchedImage.pixel(j, i);
        if (BilinearInterpolation(&image2, x, y, rgb)) {
          if (0 <= image1_x && image1_x < image1.width() &&
              0 <= image1_y && image1_y < image1.height() &&
              !(qRed(pixel1) == 0 && qGreen(pixel1) == 0 && qBlue(pixel1) == 0)) {
            // Overlapping region. Do center-weighting here.
            double weight1 = CenterWeight(image1_x, image1_y, image1.width(), image1.height(), 2);
            double weight2 = CenterWeight(static_cast<int>(x)*projected_image2_scale,
                                          static_cast<int>(y)*projected_image2_scale,
                                          image2.width()*projected_image2_scale,
                                          image2.height()*projected_image2_scale, 2);
            double total_weight = weight1 + weight2;
            // Reweight channels.
            double red = (weight1*qRed(pixel1) + weight2*rgb[0])/total_weight;
            double green = (weight1*qGreen(pixel1) + weight2*rgb[1])/total_weight;
            double blue = (weight1*qBlue(pixel1) + weight2*rgb[2])/total_weight;
            stitchedImage.setPixel(j, i, qRgb(red, green, blue));
          } else {
            // Purely image 2.
            stitchedImage.setPixel(j, i, qRgb(rgb[0], rgb[1], rgb[2]));
          }
        }
      }      
}

