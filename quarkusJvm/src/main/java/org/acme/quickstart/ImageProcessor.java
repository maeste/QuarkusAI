package org.acme.quickstart;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public final class ImageProcessor
{
   private static SameDiff sd = null;
   private static INDArray allZeros;
   static {
	   try {

	   //@SuppressWarnings("rawtypes")
       sd = TFGraphMapper.getInstance().importGraph(new File(ImageProcessor.class.getClassLoader().getResource("mobilenet_frozen.pb").getPath()));
           INDArray allZeros = Nd4j.zeros(1, 128,128,3);
       } catch (Exception e) {
		   throw new RuntimeException(e);
	   }
   }
	
   public static List<Probability> labelImage(InputStream imageStream, int numResults) throws Exception
   {
	   // Convert file to INDArray
	   NativeImageLoader loader = new NativeImageLoader(128, 128, 3);
       INDArray image = loader.asMatrix(imageStream).permute(0,2,3,1);

       //Inference returns array of INDArray, index[0] has the predictions
       sd.associateArrayWithVariable(image, sd.variableMap().get("input"));
       INDArray output = sd.execAndEndResult();
       
       // convert 1000 length numeric index of probabilities per label
       // to sorted return top numResult convert to string using helper function VGG16.decodePredictions
       // "predictions" is string of our results
       return decodePredictions(output, numResults);
   }
   
   private static List<Probability> decodePredictions(INDArray predictions, final int numResults) {
       ArrayList<String> labels = ImageNetLabels.getLabels();
       List<Probability> probs = new ArrayList<>(numResults);
       //brute force collect top numResults
       int i = 0;
//       for (int batch = 0; batch < predictions.size(0); batch++) {
//           predictionDescription += "Predictions for batch ";
//           if (predictions.size(0) > 1) {
//               predictionDescription += String.valueOf(batch);
//           }
//           predictionDescription += " :";
       final int batch = 0;
           INDArray currentBatch = predictions.getRow(batch).dup();
           while (i < numResults) {
               final int t = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
               final float f = currentBatch.getFloat(batch, t);
               currentBatch.putScalar(0, t, 0);
               probs.add(new Probability(labels.get(t), f));
               i++;
           }
//       }
       return probs;
   }

}
