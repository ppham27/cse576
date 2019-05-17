#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <thread>
#include <utility>

#include <QtGui>

#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"


/**************************************************
CODE FOR K-MEANS COLOR IMAGE CLUSTERING (RANDOM SEED)
**************************************************/

namespace {
  static constexpr int kMaxColorLevel = 256;
}

void Clustering(QImage *image, int num_clusters, int maxit) {
  int w = image->width(), h = image->height();
  QImage buffer = image->copy();

  std::vector<QRgb> centers, centers_new;

  //initialize random centers
  srand(2020);
  for (int i = 0; i < num_clusters; ++i) {
    QRgb center = qRgb(rand() % kMaxColorLevel, rand() % kMaxColorLevel, rand() % kMaxColorLevel);
    centers.push_back(center);
    centers_new.push_back(center);
  }

  //iterative part
  int it = 0;
  std::vector<int> ids;
  while (it < maxit) {
    ids.clear();
    //assign pixels to clusters
    for (int r = 0; r < h; r++)
      for (int c = 0; c < w; c++) {
        int maxd = 999999, id = 0;
        for (int n = 0; n < num_clusters; n++) {
          QRgb pcenter = centers[n];
          QRgb pnow = buffer.pixel(c, r);
          int d = abs(qRed(pcenter) - qRed(pnow)) + abs(qGreen(pcenter) - qGreen(pnow)) + abs(qBlue(pcenter) - qBlue(pnow));
          if (d < maxd) { 
            maxd = d;
            id = n;
          }
        }
        ids.push_back(id);
      }

    //update centers
    std::vector<int> cnt, rs, gs, bs;
    for (int n = 0; n < num_clusters; n++) {
      rs.push_back(0); gs.push_back(0); bs.push_back(0); cnt.push_back(0);
    }
    for (int r = 0; r < h; r++)
      for (int c = 0; c < w; c++) {
        QRgb pixel = buffer.pixel(c,r);
        rs[ids[r * w + c]] += qRed(pixel);
        gs[ids[r * w + c]] += qGreen(pixel);
        bs[ids[r * w + c]] += qBlue(pixel);
        ++cnt[ids[r * w + c]];
      }
    for (int n = 0; n < num_clusters; n++)
      if (cnt[n] == 0) // no pixels in a cluster
        continue;
      else
        centers_new[n] = qRgb(rs[n]/cnt[n], gs[n]/cnt[n], bs[n]/cnt[n]);

    centers = centers_new; it++;
  }
  //render results
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++)
      image->setPixel(c, r, qRgb(ids[r * w + c],ids[r * w + c],ids[r * w + c]));
}

/**************************************************
CODE FOR FINDING CONNECTED COMPONENTS
**************************************************/

#include "utils.h"

#define MAX_LABELS 80000

#define I(x,y)   (image[(y)*(width)+(x)])
#define N(x,y)   (nimage[(y)*(width)+(x)])

void uf_union( int x, int y, unsigned int parent[] )
{
    while ( parent[x] )
        x = parent[x];
    while ( parent[y] )
        y = parent[y];
    if ( x != y ) {
        if ( y < x ) parent[x] = y;
        else parent[y] = x;
    }
}

int uf_find(int& next_label, int x, unsigned int parent[], unsigned int label[] )
{
    while ( parent[x] )
        x = parent[x];
    if ( label[x] == 0 )
        label[x] = next_label++;
    return label[x];
}

void conrgn(int *image, int *nimage, int width, int height) {
  // TOO MANY regions for a non-main thread. Allocate memeory on the heap.
  unsigned int* parent = new unsigned int[MAX_LABELS];
  unsigned int* labels = new unsigned int[MAX_LABELS];
  int next_region = 1, k;
  int next_label = 1; // reset its value to its initial value

  memset(parent, 0, sizeof(unsigned int)*MAX_LABELS);
  memset(labels, 0, sizeof(unsigned int)*MAX_LABELS);

  for (int y = 0; y < height; ++y) {
    for ( int x = 0; x < width; ++x ) {
      k = 0;
      if ( x > 0 && I(x-1,y) == I(x,y) )
        k = N(x-1,y);
      if ( y > 0 && I(x,y-1) == I(x,y) && N(x,y-1) < k )
        k = N(x,y-1);
      if ( k == 0 ) {
        k = next_region; next_region++;
      }
      if ( k >= MAX_LABELS ) {
        fprintf(stderr, "Maximum number of labels reached. Increase MAX_LABELS and recompile.\n"); exit(1);
      }
      N(x,y) = k;
      if ( x > 0 && I(x-1,y) == I(x,y) && N(x-1,y) != k )
        uf_union( k, N(x-1,y), parent );
      if ( y > 0 && I(x,y-1) == I(x,y) && N(x,y-1) != k )
        uf_union( k, N(x,y-1), parent );
    }
  }
  for ( int i = 0; i < width*height; ++i )
    if ( nimage[i] != 0 )
      nimage[i] = uf_find(next_label, nimage[i], parent, labels );

  delete[] parent;
  delete[] labels;
}


/**************************************************
 **************************************************
TIME TO WRITE CODE
**************************************************
**************************************************/

namespace {
  std::vector<int> PreProcessRegions(const int width, const int height, const int* const region_ids,
                                     double threshold) {
    unordered_map<int, int> num_pixels_in_region;
    int num_regions = 0;
    for (int r = 0; r < height; ++r)
      for (int c = 0; c < width; ++c) {
        ++num_pixels_in_region[region_ids[r*width + c] - 1];
        num_regions = region_ids[r*width + c] > num_regions ? region_ids[r*width + c] : num_regions;
      }
    // Threshold regions by size as a percentage of the image.
    std::vector<int> remapped_region_ids; remapped_region_ids.reserve(num_regions);
    int next_remapped_id = 0;
    for (int i = 0; i < num_regions; ++i) {
      const int remapped_id =
        static_cast<double>(num_pixels_in_region[i])/(width*height) < threshold ? -1 : next_remapped_id++;
      remapped_region_ids.push_back(remapped_id);
    }
    // TODO(pmp10): Add additional code that would replace -1s and merge with one of the big regions.
    return remapped_region_ids;
  }
}

namespace {
  static constexpr int kGlcmSize = 8;
  static constexpr int kGlcmBucketSize = kMaxColorLevel / kGlcmSize;

  enum Feature {
    // Volume of region.
    kSize = 0,
    // Color features.
    kRed, kGreen, kBlue,
    // Centroid features.
    kRow, kColumn,
    // Bounding box features.
    kTop, kRight, kBottom, kLeft,
    // GLCM (gray-level co-occurence matrix).
    kGlcm,
    // Allocate enough space for GLCM.
    kNumFeatures = kGlcm + kGlcmSize*kGlcmSize,
  };

  int GetGrayLevel(const QImage& image, const int r, const int c) {
    const int pixel = image.pixel(c, r);
    const int gray_level = static_cast<int>(std::round(0.3*qRed(pixel) + 0.6*qGreen(pixel) + 0.1*qBlue(pixel)));
    return gray_level/kGlcmBucketSize;
  }

  namespace GlcmFeature {
    double Contrast(double* glcm) {
      double contrast = 0;
      for (int i = 0; i < kGlcmSize; ++i)
        for (int j = 0; j < kGlcmSize; ++j)
          contrast += (i - j)*(i - j)*glcm[i*kGlcmSize + j];
      return contrast;
    }

    double Energy(const double* const glcm) {
      return std::accumulate(glcm, glcm + kGlcmSize*kGlcmSize, 0.,
                             [](double contrast, double prob) { return contrast + prob*prob; });
    }

    double Entropy(const double* const glcm) {
      return std::accumulate(glcm, glcm + kGlcmSize*kGlcmSize, 0.,
                             [](double entropy, double prob) { return entropy + -prob*std::log2(prob); });
    }
  }  // namespace GlcmFeature
}  // namespace

/**************************************************
Code to compute the features of a given image (both database images and query image)
**************************************************/
std::vector<double*> MainWindow::ExtractFeatureVector(QImage image) {
  /********** STEP 1 **********/

  // Display the start of execution of this step in the progress box of the application window
  // You can use these 2 lines to display anything you want at any point of time while debugging

  // Perform K-means color clustering
  // This time the algorithm returns the cluster id for each pixel, not the rgb values of the corresponding cluster center
  // The code for random seed clustering is provided. You are free to use any clustering algorithm of your choice from HW 1
  // Experiment with the num_clusters and max_iterations values to get the best result

  int num_clusters = 5;
  int max_iterations = 50;
  QImage image_copy = image;
  Clustering(&image_copy, num_clusters, max_iterations);

  /********** STEP 2 **********/
  // Find connected components in the labeled segmented image
  // Code is given, you don't need to change
  const int w = image_copy.width(); const int h = image_copy.height();

  int *img = (int*) malloc(w*h * sizeof(int));
  memset( img, 0, w * h * sizeof( int ) );
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c)
      img[r*w + c] = qRed(image_copy.pixel(c, r));
  int* nimg = (int*) malloc(w*h * sizeof(int));
  memset(nimg, 0, w*h*sizeof(int));
  conrgn(img, nimg, w, h);

  const std::vector<int> remapped_region_ids = ::PreProcessRegions(h, w, nimg, /*threshold=*/0.01);
  const int num_regions = *std::max_element(remapped_region_ids.begin(), remapped_region_ids.end()) + 1;
  // The resultant image of Step 2 is 'nimg', whose values range from 1 to num_regions

  /********** STEP 3 **********/
  // Extract the feature vector of each region

  // Set length of feature vector according to the number of features you plan to use.
  featurevectorlength = kNumFeatures;

  // Initializations required to compute feature vector

  std::vector<double*> featurevector;  // final feature vector of the image; to be returned
  double** const features = new double*[num_regions];  // stores the feature vector for each connected component
  for(int i = 0; i < num_regions; ++i) {
    features[i] = new double[featurevectorlength]();  // initialize with zeros
    features[i][kTop] = h - 1; features[i][kRight] = 0; features[i][kLeft] = w - 1; features[i][kBottom] = 0;
    // Initialize GLCM with a Dirichlet prior.
    std::fill_n(features[i] + kGlcm, kGlcmSize*kGlcmSize, /*alpha=*/0.01);
  }

  // Accumulate features. We'll normalize for region size later.
  for(int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) {
      const int region_id = remapped_region_ids[nimg[r*w + c] - 1];
      if (region_id == -1) continue;
      Q_ASSERT_X(region_id < num_regions, "ExtractFeatureVector", "Out of range region ID.");
      features[region_id][kSize] += 1; // stores the number of pixels for each connected component
      // Color features.
      features[region_id][kRed] += (double) qRed(image.pixel(c, r));
      features[region_id][kGreen] += (double) qGreen(image.pixel(c, r));
      features[region_id][kBlue] += (double) qBlue(image.pixel(c, r));
      // Centroid.
      features[region_id][kRow] += static_cast<double>(r)/h;
      features[region_id][kColumn] += static_cast<double>(c)/w;
      // Bounding box.
      features[region_id][kTop] = std::min(features[region_id][kTop], static_cast<double>(r)/h);
      features[region_id][kRight] = std::max(features[region_id][kRight], static_cast<double>(c)/w);
      features[region_id][kBottom] = std::max(features[region_id][kBottom], static_cast<double>(r)/h);
      features[region_id][kLeft] = std::min(features[region_id][kLeft], static_cast<double>(c)/w);
      // GCLM with diagonal neighbor.
      const int neighbor_r = r + 1; const int neighbor_c = c + 1;
      const int neighbor_region_id =
        neighbor_r < h && neighbor_c < w ? remapped_region_ids[nimg[neighbor_r*w + neighbor_c] - 1] : -1;
      if (neighbor_region_id == region_id) {
        const int gray_level = GetGrayLevel(image, r, c);
        const int neighbor_gray_level = GetGrayLevel(image, neighbor_r, neighbor_c);
        features[region_id][kGlcm + kGlcmSize*gray_level + neighbor_gray_level] += 1;
      }
    }

  // Normalization.
  for(int region_id = 0; region_id < num_regions; ++region_id) {
    // Save the mean RGB and size values as image feature after normalization
    features[region_id][kRed] /= features[region_id][kSize]*255.0;
    features[region_id][kGreen] /= features[region_id][kSize]*255.0;
    features[region_id][kBlue] /= features[region_id][kSize]*255.0;
    // Centroid.
    features[region_id][kRow] /= features[region_id][kSize];
    features[region_id][kColumn] /= features[region_id][kSize];
    // Bounding box
    Q_ASSERT_X(features[region_id][kTop] <= features[region_id][kBottom],
               "ExtractFeatureVector", "Bounding box is invalid.");
    Q_ASSERT_X(features[region_id][kLeft] <= features[region_id][kRight],
               "ExtractFeatureVector", "Bounding box is invalid.");
    // GLCM, make a joint probability distribution
    const int num_glcm_observations =
      std::accumulate(features[region_id] + kGlcm, features[region_id] + kGlcm + kGlcmSize*kGlcmSize, 0);
    std::transform(features[region_id] + kGlcm,
                   features[region_id] + kGlcm + kGlcmSize*kGlcmSize,
                   features[region_id] + kGlcm, [n = num_glcm_observations](double x) { return x/n; });
    // Normalize volume and save.
    features[region_id][0] /= (double) w*h;
    featurevector.push_back(features[region_id]);
  }

  // Return the created feature vector
  return featurevector;
}


/***** Code to compute the distance between two images *****/

namespace {
  // Function that implements distance measure 1
  double distance1(const double* vector1, const double* vector2, int length) {
    // default, for trial only; change according to your distance measure
    double distance = 0;
    for (int i = 0; i < length; ++i) {
      double delta = vector1[i] - vector2[i];
      distance += delta*delta;
    }
    return distance;
  }

  // Function that implements distance measure 2
  double distance2(const double* vector1, const double* vector2, int length) {
    // default, for trial only; change according to your distance measure
    return ((double) rand() / (double) RAND_MAX);
  }
}

// Function to calculate the distance between two images
// Input argument isOne takes true for distance measure 1 and takes false for distance measure 2

void MainWindow::CalculateDistances(bool isOne) {
  const std::function<double(const std::vector<double*>&,
                             const std::vector<double*>&,
                             const std::function<double(const double*, const double*)>&)> match_region_by_distance =
    [](const std::vector<double*>& query_features,
       const std::vector<double*>& image_regions,
       const std::function<double(const double*, const double*)>& distance_fn) -> double {
    double distance = 0.;
    for (const double* const query_feature : query_features) {  // for each region in the query image
      double min_region_distance = std::numeric_limits<double>::max();  // distance between the best matching regions
      for (const double* const region_feature : image_regions)  // for each region in the current database image
        min_region_distance = std::min(min_region_distance, distance_fn(query_feature, region_feature));
      distance += min_region_distance;  // sum of distances between each matching pair of regions
    }
    return distance/query_features.size();  // normalize by number of matching pairs
  };

  const std::function<double(const double*, const double*)> distance_fn =
    std::bind(isOne ? ::distance1 : ::distance2,
              std::placeholders::_1, std::placeholders::_2, featurevectorlength);

  // Initialize distances to max.
  std::fill_n(this->distances, num_images, std::numeric_limits<double>::max());

  // Put calculated distances in queue for further processing.
  std::queue<std::pair<int, double>> distances; std::mutex distances_mutex;
  const std::function<void(const std::vector<double*>&, int)>
    enqueue_distance = [&database = this->databasefeatures,
                        &match_region_by_distance,
                        &distance_fn,
                        &distances,
                        &distances_mutex](const std::vector<double*>& query_features, int i) {
    double distance = match_region_by_distance(query_features, database[i], distance_fn);
    std::lock_guard<std::mutex> lock(distances_mutex);
    distances.emplace(i, distance);
  };

  const size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads; threads.reserve(num_threads);
  for(int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back([&enqueue_distance](const std::vector<double*>& query_features,
                                             int start, int stop, int step) {
                           for (int i = start; i < stop; i += step) enqueue_distance(query_features, i);
                         }, queryfeature, thread_id, num_images, num_threads);
  }
  
  int num_distances_computed = 0;
  while (num_distances_computed < num_images) {
    if (distances.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    while (!distances.empty()) {
      std::pair<int, double> distance = distances.front(); distances.pop();
      this->distances[distance.first] = distance.second;  // Store distances.
      // Display the distance values.
      ui->progressBox->append(QString::fromStdString("Distance to image " + std::to_string(distance.first+1) +
                                                     " = " + std::to_string(distance.second)));
      ++num_distances_computed;
    }
    QApplication::processEvents();
  }
  for (std::thread& t : threads) t.join();
}
