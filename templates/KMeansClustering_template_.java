import ij.*;
import ij.plugin.filter.PlugInFilter;
import ij.process.*;

import java.util.Arrays;
import java.util.Random;
import java.util.Vector;
import ij.gui.GenericDialog;

public class KMeansClustering_ implements PlugInFilter {

    public int setup(String arg, ImagePlus imp) {
        if (arg.equals("about"))
        {showAbout(); return DONE;}
        return DOES_RGB;
    } //setup


    public void run(ImageProcessor ip) {
        double[] blackCluster = new double[] {0, 0, 0};
        double[] redCluster = new double[] {255, 0, 0};
        double[] blueCluster = new double[] {0, 0, 255};
        double[] greenCluster = new double[] {0, 255, 0};
        Vector<double[]> clusterCentroides = new Vector<double[]>();
        clusterCentroides.add(blackCluster);
        clusterCentroides.add(redCluster);
        clusterCentroides.add(greenCluster);
        clusterCentroides.add(blueCluster);

        
        int numOfIterations = 15;

        int width = ip.getWidth();
        int height = ip.getHeight();

        //input image ==> 3D
        int[][][] inImgRGB =ImageJUtility.getChannelImageFromIP(ip, width, height, 3);

        for(int i = 0; i < numOfIterations; i++) {
            System.out.println("cluster update # " + i);
            clusterCentroides = UpdateClusters(inImgRGB, clusterCentroides, width, height);
        }

        int[][][] resImgRGB = new int[width][height][3];
        
		//TODO: implementation required

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
		//TODO: implementation required
		
		
        return null;
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

        //TODO: implementation required

        return minClusterIDX;
    }


    void showAbout() {
        IJ.showMessage("About KMeansClustering_...",
                "this is a PluginFilter to segment RGB input images in an automated way\n");
    } //showAbout

} //class KMeansClustering_