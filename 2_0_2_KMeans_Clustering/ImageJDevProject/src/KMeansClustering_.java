import ij.*;
import ij.plugin.filter.PlugInFilter;
import ij.process.*;

import java.util.Vector;

public class KMeansClustering_ implements PlugInFilter {
 
	public int setup(String arg, ImagePlus imp) {
		if (arg.equals("about"))
			{showAbout(); return DONE;}
		return DOES_RGB;
	} //setup

	
	public void run(ImageProcessor ip) {

		int width = ip.getWidth();
		int height = ip.getHeight();
        int[][][] inImgRGB = ImageJUtility.getChannelImageFromIP(
				ip, width, height, 3);

		//let the user define the cluster
		Vector<double[]> clusterCentroids = new Vector<>();
		clusterCentroids.add(new double[]{0, 0, 255}); //blue
		clusterCentroids.add(new double[]{255, 0, 0}); //red
		clusterCentroids.add(new double[]{0, 0, 255}); //green
		clusterCentroids.add(new double[]{255, 255, 255}); //white
		clusterCentroids.add(new double[]{0, 0, 0}); //black

		int numOfIterations = 5;

		//first assign pixels to clusters
		//clusters should not be empty ==> DIV 0

		for(int i = 0; i < numOfIterations; i++) {
			System.out.println("cluster update # " + i);
			clusterCentroids = UpdateClusters(inImgRGB, clusterCentroids, width, height);
		}

		int[][][] resImgRGB = new int[width][height][3];
		//define the final colors to draw (== number of clusters)
		Vector<int[]> intValRGB = new Vector<>();
		for(double[] dblValRGB : clusterCentroids) {
			intValRGB.add(new int[]{(int)Math.round(dblValRGB[0]),
					                (int)Math.round(dblValRGB[1]),
					                (int)Math.round(dblValRGB[2])});
		} //for convert color from double*3 to int*3

		//finally get the result image and color the pixels
		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {
              int closestClusterID = GetBestClusterIdx(inImgRGB[x][y], clusterCentroids);
			  resImgRGB[x][y] = intValRGB.get(closestClusterID);
			} //for all x
		} //for all x


		ImageJUtility.showNewImageRGB(resImgRGB, width, height,
				"final segmented image with centroid colors");
        
	} //run

	/*
    iterate all pixel and assign them to the cluster showing the smallest distance
    then, for each color centroid, the average color (RGB) gets update
     */
	Vector<double[]> UpdateClusters(int[][][] inRGBimg, Vector<double[]> inClusters, int width, int height) {
		//allocate the data structures
		double[][] newClusterMeanSumArr = new double[inClusters.size()][3]; //for all clusters, the sum for R, G and B
		int[] clusterCountArr = new int[inClusters.size()];

		//process all pixels
		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {
				int[] currRGB = inRGBimg[x][y];
				int bestClusterIdx = GetBestClusterIdx(currRGB, inClusters);
				clusterCountArr[bestClusterIdx]++;
				newClusterMeanSumArr[bestClusterIdx][0] += currRGB[0];
				newClusterMeanSumArr[bestClusterIdx][1] += currRGB[1];
				newClusterMeanSumArr[bestClusterIdx][2] += currRGB[2];
			}
		}

		//finally calc the new centroids
		Vector<double[]> outClusters = new Vector<>();
        int clusterIdx = 0;
		for (double[] clusterCentroid : inClusters) {
			double[] newClusterColor = newClusterMeanSumArr[clusterIdx];
			int numOfElements = clusterCountArr[clusterIdx];

			if(numOfElements > 0) {
				newClusterColor[0] /= numOfElements;
				newClusterColor[1] /= numOfElements;
				newClusterColor[2] /= numOfElements;
				outClusters.add(newClusterColor);
			} else {
				outClusters.add(clusterCentroid); //fallback if empty, take the old one
			}

			clusterIdx++;
		}

		return outClusters;
	}

	double ColorDist(double[] refColor, int[] currColor) {
		double diffR = refColor[0] - currColor[0];
		double diffG = refColor[1] - currColor[1];
		double diffB = refColor[2] - currColor[2];

		double resDist = Math.sqrt(diffR * diffR + diffG * diffG + diffB * diffB);
		return  resDist;
	}

	/*
    returns the cluster IDX showing min distance to input pixel
     */
	int GetBestClusterIdx(int[] rgbArr, Vector<double[]> clusters) {
		double minDist = ColorDist(clusters.get(0), rgbArr);
		int minClusterIDX = 0;

		for(int currClusterIDX = 1; currClusterIDX < clusters.size(); currClusterIDX++) {
			double currDist = ColorDist(clusters.get(currClusterIDX), rgbArr);
			if(currDist < minDist) {
				minDist = currDist;
				minClusterIDX = currClusterIDX;
			} //if new best found
		}
		return minClusterIDX;
	}

	void showAbout() {
		IJ.showMessage("About KMeansClustering_...",
			"this is a PluginFilter for clustering RGB images \n");
	} //showAbout
	
} //class KMeansClustering_

