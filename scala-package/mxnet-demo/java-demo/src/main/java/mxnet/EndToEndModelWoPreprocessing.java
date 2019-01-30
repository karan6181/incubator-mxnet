/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mxnet;

import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.*;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.LongStream;


/**
 * This Class is a demo to show how users can use Predictor APIs to do
 * Image Classification with all hand-crafted Pre-processing.
 * All helper functions for image pre-processing are
 * currently available in ObjectDetector class.
 */
public class EndToEndModelWoPreprocessing {
    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = "/model/ssd_resnet50_512";
    @Option(name = "--input-image", usage = "the input image")
    private String inputImagePath = "/images/dog.jpg";
    @Option(name = "--input-dir", usage = "the input batch of images directory")
    private String inputImageDir = "/images/";
    @Option(name = "--num-runs", usage = "Number of runs")
    private String numRuns = "1";
    @Option(name = "--batchsize", usage = "batch size")
    private String batchsize = "1";

    final static Logger logger = LoggerFactory.getLogger(EndToEndModelWoPreprocessing.class);

    /**
     * Load the image from file to buffered image
     * It can be replaced by loadImageFromFile from ObjectDetector
     * @param inputImagePath input image Path in String
     * @return Buffered image
     */
    private static BufferedImage loadImageFromFile(String inputImagePath) {
        BufferedImage buf = null;
        try {
            buf = ImageIO.read(new File(inputImagePath));
        } catch (IOException e) {
            System.err.println(e);
        }
        return buf;
    }

    /**
     * Load the images from batch of files
     * @param batchFile List of image file path
     * @return List of Buffered image
     */
    private static List<BufferedImage> loadInputBatch(List<String> batchFile) {
        List<BufferedImage> listBufferedImage = new ArrayList<>();

        for (String filename : batchFile) {
            listBufferedImage.add(loadImageFromFile(filename));
        }
        return listBufferedImage;
    }

    /**
     * Reshape the current image using ImageIO and Graph2D
     * It can be replaced by reshapeImage from ObjectDetector
     * @param buf Buffered image
     * @param newWidth desired width
     * @param newHeight desired height
     * @return a reshaped bufferedImage
     */
    private static BufferedImage reshapeImage(BufferedImage buf, int newWidth, int newHeight) {
        BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(buf, 0, 0, newWidth, newHeight, null);
        g.dispose();
        return resizedImage;
    }

    /**
     * Generate the List of batches of input image file path.
     * @param inputImageDirPath Input image directory
     * @param batchSize Batch size
     * @return List of batches
     */
    private static List<List<String>> generateBatches(String inputImageDirPath, int batchSize) {
        File dir = new File(inputImageDirPath);

        List<List<String>> output = new ArrayList<List<String>>();
        List<String> batch = new ArrayList<String>();
        for (File imgFile : dir.listFiles()) {
            batch.add(imgFile.getPath());
            if (batch.size() == batchSize) {
                output.add(batch);
                batch = new ArrayList<String>();
            }
        }
        if (batch.size() > 0) {
            output.add(batch);
        }
        return output;
    }

    /**
     * Convert an image from a buffered image into pixels float array
     * It can be replaced by bufferedImageToPixels from ObjectDetector
     * @param buf buffered image
     * @return Float array
     */
    private static float[] imagePreprocess(BufferedImage buf) {
        // Get height and width of the image
        int w = buf.getWidth();
        int h = buf.getHeight();

        // get an array of integer pixels in the default RGB color mode
        int[] pixels = buf.getRGB(0, 0, w, h, null, 0, w);

        // 3 times height and width for R,G,B channels
        float[] result = new float[3 * h * w];

        int row = 0;
        // copy pixels to array vertically
        while (row < h) {
            int col = 0;
            // copy pixels to array horizontally
            while (col < w) {
                int rgb = pixels[row * w + col];
                // getting red color
                result[0 * h * w + row * w + col] = ((rgb >> 16) & 0xFF) / 255.0f; // 0
                // getting green color
                result[1 * h * w + row * w + col] = ((rgb >> 8) & 0xFF) / 255.0f; // 50176
                // getting blue color
                result[2 * h * w + row * w + col] = (rgb & 0xFF) / 255.0f; // 100352
                col += 1;
            }
            row += 1;
        }
        buf.flush();
        float[] norm_image = normalizeImage(result);
        return norm_image;
    }

    /**
     * Normalize the image tensor with mean and standard deviation.
     * @param img Float array
     * @return Normalized Float array
     */
    private static float[] normalizeImage(float[] img) {
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        int w = 224;
        int h = 224;

        int row = 0;
        // copy pixels to array vertically
        while (row < h) {
            int col = 0;
            // copy pixels to array horizontally
            while (col < w) {
                // getting red color
                img[0 * h * w + row * w + col] = (img[0 * h * w + row * w + col] - mean[0]) / std[0];
                // getting green color
                img[1 * h * w + row * w + col] = (img[1 * h * w + row * w + col] - mean[1]) / std[1];
                // getting blue color
                img[2 * h * w + row * w + col] = (img[2 * h * w + row * w + col] - mean[2]) / std[2];
                col += 1;
            }
            row += 1;
        }
        return img;
    }

    private static long percentile(int p, long[] seq) {
        Arrays.sort(seq);
        int k = (int) Math.ceil((seq.length - 1) * (p / 100.0));
        return seq[k];
    }

    private static void printStatistics(long[] inferenceTimesRaw, String metricsPrefix)  {
        long[] inferenceTimes = inferenceTimesRaw;
        // remove head and tail
        if (inferenceTimes.length > 2) {
            inferenceTimes = Arrays.copyOfRange(inferenceTimesRaw,
                    1, inferenceTimesRaw.length - 1);
        }
        double p50 = percentile(50, inferenceTimes) / 1.0e6;
        double p99 = percentile(99, inferenceTimes) / 1.0e6;
        double p90 = percentile(90, inferenceTimes) / 1.0e6;
        long sum = 0;
        for (long time: inferenceTimes) sum += time;
        double average = sum / (inferenceTimes.length * 1.0e6);

        System.out.println(
                String.format("\n%s_p99 %fms\n%s_p90 %fms\n%s_p50 %fms\n%s_average %1.2fms",
                        metricsPrefix, p99, metricsPrefix, p90,
                        metricsPrefix, p50, metricsPrefix, average)
        );

    }

    /**
     * Helper class to print the maximum prediction result
     * @param probabilities The float array of probability
     * @param modelPathPrefix model Path needs to load the synset.txt
     */
    private static String printMaximumClass(float[] probabilities,
                                            String modelPathPrefix) throws IOException {
        String synsetFilePath = modelPathPrefix.substring(0,
                1 + modelPathPrefix.lastIndexOf(File.separator)) + "/synset.txt";
        BufferedReader reader = new BufferedReader(new FileReader(synsetFilePath));
        ArrayList<String> list = new ArrayList<>();
        String line = reader.readLine();

        while (line != null){
            list.add(line);
            line = reader.readLine();
        }
        reader.close();

        int maxIdx = 0;
        for (int i = 1;i<probabilities.length;i++) {
            if (probabilities[i] > probabilities[maxIdx]) {
                maxIdx = i;
            }
        }

        return "Probability : " + probabilities[maxIdx] + " Class : " + list.get(maxIdx) ;
    }

    public static void main(String[] args) throws IOException {
        EndToEndModelWoPreprocessing inst = new EndToEndModelWoPreprocessing();
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            parser.printUsage(System.err);
            System.exit(1);
        }

	boolean useBatch = false;
	int batchSize = Integer.parseInt(inst.batchsize);
        // int batchSize = 10;
	int numberOfRuns = Integer.parseInt(inst.numRuns);
        // int numberOfRuns = 1;
        String imgPath = inst.inputImagePath;
        String imgDir = inst.inputImageDir;


        // Prepare the model
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }

        Shape inputShape = new Shape(new int[]{1, 3, 224, 224});
        List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        Predictor predictor = new Predictor(inst.modelPathPrefix, inputDescriptors, context,0);


        if(useBatch) {

            // Loading batch of images from the directory path
            List<List<String>> batchFiles = generateBatches(imgDir, batchSize);
            long[] infTime = new long[numberOfRuns];
            long[] batchTime = new long[batchFiles.size()];
            for (int i = 0; i < numberOfRuns; i++) {
                for (int j = 0; j < batchFiles.size(); j++) {
                    List<BufferedImage> imgList = loadInputBatch(batchFiles.get(j));
                    NDArray[] array = new NDArray[imgList.size()];
                    List<NDArray> listImageTensor = new ArrayList<>();
                    for (int k = 0; k < imgList.size(); k++) {
                        BufferedImage img = reshapeImage(imgList.get(k), 224, 224);
                        NDArray nd = new NDArray(
                                imagePreprocess(img), inputShape, Context.cpu());
                        array[k] = nd;
                    }

                    NDArray[] arr2 = NDArray.concat(array, array.length, 0, null);
                    listImageTensor.add(arr2[0]);
                    arr2[0].waitToRead();

                    long currTime = System.nanoTime();
                    List<NDArray> res = predictor.predictWithNDArray(listImageTensor);
                    res.get(0).waitToRead();
                    batchTime[i] = System.nanoTime() - currTime;
                    System.out.println("Inference time at iteration: " + i + " and batch: " + j + " is : " + (batchTime[i] / 1.0e6) + "\n");
                }
                infTime[i] = LongStream.of(batchTime).sum() / batchTime.length;
            }
            printStatistics(infTime, "batch_inference");

        } else {
            BufferedImage bImg = loadImageFromFile(imgPath);
            bImg = reshapeImage(bImg, 224, 224);
            float[] image_tensor = imagePreprocess(bImg);
            long[] times = new long[numberOfRuns];

            for (int i = 0; i < numberOfRuns; i++) {
                long currTime = System.nanoTime();
                float[][] result = predictor.predict(new float[][]{image_tensor});

                times[i] = System.nanoTime() - currTime;
                try {
                    System.out.println("Inference time at iteration: " + i + " is : " + (times[i] / 1.0e6) + "\n");
                    System.out.println(printMaximumClass(result[0], inst.modelPathPrefix));
                } catch (IOException e) {
                    System.err.println(e);
                }
            }
            printStatistics(times, "single_inference");
        }
    }

}
