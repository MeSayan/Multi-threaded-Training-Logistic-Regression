/*
Name : Assignment 7
Creator : Sayan Mahapatra 
Date : 03-11-2021
*/

/*Dataset: CovType Binary-
- Link:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2
- No of Features: 54
- No of data points: 581012
*/

#include <iostream>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>

using namespace std;
typedef vector<double> VEC;         // vector
typedef vector<vector<double>> MAT; // features

#define EPOCHS 10               // no of epochs to train
#define NDIM 54                 // no of dimensions in feature
#define NDTP 581012             // no of data points
#define BATCHSIZE 5120          // no of data points in a batch
#define MAX_STRING_LENGTH 2000  // max string length of a line
#define NWORKERS 4              // no of update threads
#define INITIAL_LEARNING_RATE 1 // initial learning rate

// Change as per data
#define FILENAME "data" // file name of data set
#define LABEL_0 1       // label of class 0 in the file
#define LABEL_1 2       // label of class 1 in the file

// Stores the full test data set
VEC Y_test;
MAT X_test;

pthread_t threads[NWORKERS];
pthread_attr_t attr;

pthread_cond_t reader_cv, worker_cv;
pthread_mutex_t read_mutex, work_mutex;

int batches_read;
int n_batches;
pthread_mutex_t batches_read_mutex;

// Model parameters
VEC WEIGHTS(NDIM + 1, 0);
double bias = 0;
VEC DEL_WEIGHTS(NDIM + 1, 0);
double del_bias = 0;

// termination flag for workers
int terminate_flags[NWORKERS];

pthread_mutex_t update_mutex;
pthread_cond_t update_cv;

// Returns sigma(x)
double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }

// Protect access to file
FILE *fp;
pthread_mutex_t file_access_mutex;
pthread_mutex_t weight_access_mutex;

void *worker(void *arg) {
    int id = *(int *)arg;
    MAT X;
    VEC Y;
    char linebuf[MAX_STRING_LENGTH];
    while (!terminate_flags[id]) {
        while (batches_read < n_batches && !terminate_flags[id]) {
            pthread_mutex_lock(&file_access_mutex);
            // Always check for termination on wakeup
            if (terminate_flags[id]) {
                pthread_mutex_unlock(&file_access_mutex);
                break;
            }

            // Read Data from file
            if (!X.empty())
                X.clear();
            if (!Y.empty())
                Y.clear();
            int label, index;
            double val;

            // Read BATCHSIZE lines
            for (int i = 0; i < BATCHSIZE; ++i) {
                // Reset before read
                label = index = -2;
                val = -1;

                // Read a line
                char *ret = fgets(linebuf, MAX_STRING_LENGTH, fp);
                // EOF
                if (ret == NULL || strlen(ret) < 2) {
                    break;
                }

                VEC x(NDIM + 1, 0);

                char *p = strtok(linebuf, " ");
                sscanf(p, "%d", &label);
                if (label == LABEL_0)
                    Y.push_back(0);
                else if (label == LABEL_1)
                    Y.push_back(1);
                else
                    continue; // skip row as invalid label

                while (p) {
                    p = strtok(NULL, " ");
                    if (!p)
                        break;
                    if (strlen(p) == 1)
                        break;
                    sscanf(p, "%d:%lf", &index, &val);
                    x[index] = val;
                }
                X.push_back(x);
            }

            pthread_mutex_unlock(&file_access_mutex);

            if (terminate_flags[id])
                break;

            // Process the data read
            int n_samples = (int)X.size();
            VEC Y_PRED(n_samples, 0);

            for (int i = 0; i < n_samples; ++i) {
                // Calculate w.x+b
                double ans = 0;
                for (int j = 1; j <= NDIM; ++j) {
                    ans += WEIGHTS[j] * X[i][j];
                }
                ans += bias;
                // Calculate Y_PRED
                Y_PRED[i] = sigmoid(ans);
            }

            pthread_mutex_lock(&update_mutex);

            // Terminate if needed after wakeup
            if (terminate_flags[id]) {
                pthread_mutex_unlock(&file_access_mutex);
                break;
            }

            // Reset Del_Weights, Del Bias
            for (int i = 0; i < NDIM + 1; ++i)
                DEL_WEIGHTS[i] = 0;
            del_bias = 0;
            // Compute Del_Weights, Del_Bias
            for (int i = 0; i < n_samples; ++i) {
                del_bias += (Y_PRED[i] - Y[i]);
                for (int j = 1; j <= NDIM; ++j) {
                    double dd = DEL_WEIGHTS[j];
                    DEL_WEIGHTS[j] += (Y_PRED[i] - Y[i]) * X[i][j];
                }
            }

            // Send update to main thread
            pthread_cond_signal(&update_cv);
            pthread_mutex_unlock(&update_mutex);
        }
    }

    pthread_exit(NULL);
}

// Sets termination flags for all threads so that they stop
// called by main when training finishes
void stop_all_threads() {
    // Mark all process for deletion
    for (int i = 0; i < NWORKERS; ++i)
        terminate_flags[i] = 1;

    // Wake up all waiting processes
    pthread_mutex_lock(&update_mutex);
    pthread_cond_broadcast(&update_cv);
    pthread_mutex_unlock(&update_mutex);

    // Cleanup and desroy shared resources
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&batches_read_mutex);
    pthread_mutex_destroy(&update_mutex);
    pthread_cond_destroy(&update_cv);
    pthread_mutex_destroy(&file_access_mutex);
}

// Loads the entire file into X_test and Y_test for model evaluation
void load_test_data() {
    char linebuf[MAX_STRING_LENGTH];
    FILE *fp2 = fopen(FILENAME, "r");
    for (int i = 0; i < NDTP; ++i) {
        int label, index;
        double val;
        label = index = -2;
        val = -1;
        char *ret = fgets(linebuf, MAX_STRING_LENGTH, fp2);
        VEC x_test(NDIM + 1, 0);
        char *p = strtok(linebuf, " ");
        sscanf(p, "%d", &label);
        if (label == LABEL_0)
            Y_test.push_back(0);
        else if (label == LABEL_1)
            Y_test.push_back(1);
        else
            continue; // skip invalid label
        while (p) {
            p = strtok(NULL, " ");
            if (!p)
                break;
            if (strlen(p) == 1)
                break;
            sscanf(p, "%d:%lf", &index, &val);
            x_test[index] = val;
        }
        X_test.push_back(x_test);
    }
    fclose(fp2);
}

// Evaluates the model and outputs accuracy
void evaluate_model() {
    cout << "Evaluating Model " << endl;
    VEC Y_pred(NDTP, 0);
    int c, w;
    c = w = 0;
    for (int i = 0; i < NDTP; ++i) {
        // Calculate w.x+b
        double ans = 0;
        for (int j = 1; j <= NDIM; ++j)
            ans += WEIGHTS[j] * X_test[i][j];
        ans += bias;
        // Calculate Y_PRED
        Y_pred[i] = sigmoid(ans);
        if (Y_test[i] == 0 && Y_pred[i] < 0.5)
            c++;
        else if (Y_test[i] == 1 && Y_pred[i] >= 0.5)
            c++;
        else
            w++;
    }
    cout << "Model Accuracy " << 100.0 * c / (c + w) << " %" << endl;
}

// Display Model parameters (weights and bias)
void display_weights() {
    // Print Weights
    cout << "\n\n++ Model Parameters ++" << endl;
    cout << "Weights " << endl;
    for (int j = 1; j <= NDIM; ++j) {
        cout << WEIGHTS[j] << endl;
    }
    cout << "Bias " << endl << bias << endl;
}

int main(int argc, char *argv[]) {

    batches_read = 0;
    // calculate no of batches
    n_batches = (int)ceil(1.0 * NDTP / BATCHSIZE);

    // initialise termination flag
    for (int i = 0; i < 1 + NWORKERS; ++i)
        terminate_flags[i] = 0;

    // Assign unqiue id to each worker
    int worker_ids[NWORKERS];
    for (int i = 0; i < NWORKERS; ++i)
        worker_ids[i] = i;

    // load test data
    load_test_data();

    pthread_mutex_init(&read_mutex, NULL);
    pthread_cond_init(&reader_cv, NULL);
    pthread_mutex_init(&work_mutex, NULL);
    pthread_cond_init(&worker_cv, NULL);
    pthread_mutex_init(&batches_read_mutex, NULL);
    pthread_mutex_init(&update_mutex, NULL);
    pthread_cond_init(&update_cv, NULL);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    cout << "++ Training Details ++" << endl;
    cout << "No of Epochs(T): " << EPOCHS << endl;
    cout << "No of Update Threads(K): " << NWORKERS << endl;
    cout << "No of datapoints: " << X_test.size() << endl;
    cout << "No of features: " << NDIM << endl;
    cout << "Batch Size: " << BATCHSIZE << endl;

    // Open file
    fp = fopen(FILENAME, "r");

    // Start Worker Threads
    for (int w = 0; w < NWORKERS; ++w) {
        pthread_create(&threads[w], &attr, worker, (void *)&worker_ids[w]);
    }

    time_t start = time(NULL);
    cout << "\n++ Training Model ++" << endl;
    // Wait for updates
    int t = 1;
    double lr0 = INITIAL_LEARNING_RATE;

    for (int t = 1; t <= EPOCHS; ++t) {
        double lr;
        // learning rate for this epoch
        lr = lr0 / sqrt(t);
        cout << "\nEpoch : " << t << endl;
        while (1) {
            pthread_mutex_lock(&update_mutex);
            // Wait for an update to come
            pthread_cond_wait(&update_cv, &update_mutex);
            // Apply update
            bias -= lr * del_bias;
            for (int j = 1; j <= NDIM; ++j) {
                WEIGHTS[j] -= lr * DEL_WEIGHTS[j];
            }

            // Unlock (allow further updates)
            pthread_mutex_unlock(&update_mutex);
            // Mark batch as read
            pthread_mutex_lock(&batches_read_mutex);
            batches_read++;
            if (batches_read == n_batches) {
                // End of epoch
                if (t != EPOCHS) {
                    batches_read = 0;
                    fclose(fp);
                    // Reopen file for next epoch
                    fp = fopen(FILENAME, "r");
                }
                pthread_mutex_unlock(&batches_read_mutex);
                break;
            }
            pthread_mutex_unlock(&batches_read_mutex);
        }
        // Evaluate model in every epoch
        evaluate_model();
    }

    cout << "\n++ Training Finished ++" << endl;
    time_t end = time(NULL);
    cout << "Training took " << difftime(end, start) << " seconds" << endl;
    stop_all_threads();
    cout << "\n++ Stopped all threads ++" << endl;
    display_weights();
    // join all threads
    for (int i = 0; i < NWORKERS; ++i) {
        pthread_join(threads[i], NULL);
    }
    pthread_exit(NULL);
    return 0;
}
