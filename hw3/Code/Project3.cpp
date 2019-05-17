#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <limits>
#include <queue>
#include <thread>
#include <utility>

#include <QtGui>

#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"


/**************************************************
CODE FOR K-MEANS COLOR IMAGE CLUSTERING (RANDOM SEED)
**************************************************/

void Clustering(QImage *image, int num_clusters, int maxit)
{
        int w = image->width(), h = image->height();
        QImage buffer = image->copy();

        std::vector<QRgb> centers, centers_new;

        //initialize random centers
        int n = 1; srand(2020);
        while (n <= num_clusters)
        {
            QRgb center = qRgb(rand() % 256, rand() % 256, rand() % 256);
            centers.push_back(center);
            centers_new.push_back(center);
            n++;
        }

        //iterative part
        int it = 0;
        std::vector<int> ids;
        while (it < maxit)
        {
                ids.clear();
                //assign pixels to clusters
                for (int r = 0; r < h; r++)
                	for (int c = 0; c < w; c++)
                	{
                        int maxd = 999999, id = 0;
                        for (int n = 0; n < num_clusters; n++)
                        {
                                QRgb pcenter = centers[n];
                                QRgb pnow = buffer.pixel(c, r);
                                int d = abs(qRed(pcenter) - qRed(pnow)) + abs(qGreen(pcenter) - qGreen(pnow)) + abs(qBlue(pcenter) - qBlue(pnow));
                                if (d < maxd)
                                {
                                        maxd = d; id = n;
                                }
                        }
                        ids.push_back(id);
                	}

                //update centers
                std::vector<int> cnt, rs, gs, bs;
                for (int n = 0; n < num_clusters; n++)
                {
                        rs.push_back(0); gs.push_back(0); bs.push_back(0); cnt.push_back(0);
                }
                for (int r = 0; r < h; r++)
                    for (int c = 0; c < w; c++)
                    {
                        QRgb pixel = buffer.pixel(c,r);
                        rs[ids[r * w + c]] += qRed(pixel);
                        gs[ids[r * w + c]] += qGreen(pixel);
                        bs[ids[r * w + c]] += qBlue(pixel);
                        cnt[ids[r * w + c]]++;
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
  return;
}


/**************************************************
 **************************************************
TIME TO WRITE CODE
**************************************************
**************************************************/


/**************************************************
Code to compute the features of a given image (both database images and query image)
**************************************************/
std::vector<double*> MainWindow::ExtractFeatureVector(QImage image)
{
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
    Clustering(&image_copy,num_clusters,max_iterations);


    /********** STEP 2 **********/
    // Find connected components in the labeled segmented image
    // Code is given, you don't need to change

    int r, c, w = image_copy.width(), h = image_copy.height();
    int *img = (int*)malloc(w*h * sizeof(int));
    memset( img, 0, w * h * sizeof( int ) );
    int *nimg = (int*)malloc(w*h *sizeof(int));
    memset( nimg, 0, w * h * sizeof( int ) );

    for (r=0; r<h; r++)
        for (c=0; c<w; c++)
            img[r*w + c] = qRed(image_copy.pixel(c,r));

    conrgn(img, nimg, w, h);

    int num_regions=0;
    for (r=0; r<h; r++)
        for (c=0; c<w; c++)
            num_regions = (nimg[r*w+c]>num_regions)? nimg[r*w+c]: num_regions;
    // The resultant image of Step 2 is 'nimg', whose values range from 1 to num_regions

    // WRITE YOUR REGION THRESHOLDING AND REFINEMENT CODE HERE


    /********** STEP 3 **********/
    // Extract the feature vector of each region

    // Set length of feature vector according to the number of features you plan to use.
    featurevectorlength = 4;

    // Initializations required to compute feature vector

    std::vector<double*> featurevector; // final feature vector of the image; to be returned
    double **features = new double* [num_regions]; // stores the feature vector for each connected component
    for(int i=0;i<num_regions; i++)
        features[i] = new double[featurevectorlength](); // initialize with zeros

    // Sample code for computing the mean RGB values and size of each connected component

    for(int r=0; r<h; r++)
      for (int c=0; c<w; c++) {
        features[nimg[r*w+c]-1][0] += 1; // stores the number of pixels for each connected component
        features[nimg[r*w+c]-1][1] += (double) qRed(image.pixel(c,r));
        features[nimg[r*w+c]-1][2] += (double) qGreen(image.pixel(c,r));
        features[nimg[r*w+c]-1][3] += (double) qBlue(image.pixel(c,r));
      }

    // Save the mean RGB and size values as image feature after normalization
    for(int m=0; m<num_regions; m++) {
      for(int n=1; n<featurevectorlength; n++) features[m][n] /= features[m][0]*255.0;
      features[m][0] /= (double) w*h;
      featurevector.push_back(features[m]);
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

  // Put calculated distances in queue for furhter processing.
  std::queue<std::pair<int, double>> distances;
  const size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads; threads.reserve(num_threads);
  for(int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back([&match_region_by_distance,
                          &distance_fn](const std::vector<double*>& query_features,
                                        const std::vector<std::vector<double*>>& database,
                                        int start, int stop, int step,
                                        std::queue<std::pair<int, double>>* distances) {
                           for (int i = start; i < stop; i += step)
                             distances->emplace(i, match_region_by_distance(query_features, database[i], distance_fn));
                         }, queryfeature, databasefeatures, thread_id, num_images, num_threads, &distances);
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
