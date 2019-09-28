package org.acme.quickstart;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public final class ImageProcessor
{
   private static ComputationGraph vgg16 = null;
   static {
	   try {
	   @SuppressWarnings("rawtypes")
	   ZooModel zooModel = VGG16.builder().build();
	   vgg16 = (ComputationGraph)zooModel.initPretrained(PretrainedType.IMAGENET);
	   } catch (Exception e) {
		   throw new RuntimeException(e);
	   }
   }
	
   public static List<Probability> labelImage(InputStream imageStream, int numResults) throws Exception
   {
	   // Convert file to INDArray
	   NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
       INDArray image = loader.asMatrix(imageStream);
       
       // Mean subtraction pre-processing step for VGG
       DataNormalization scaler = new VGG16ImagePreProcessor();
       scaler.transform(image);

       //Inference returns array of INDArray, index[0] has the predictions
       INDArray[] output = vgg16.output(false,image);
       
       // convert 1000 length numeric index of probabilities per label
       // to sorted return top numResult convert to string using helper function VGG16.decodePredictions
       // "predictions" is string of our results
       return decodePredictions(output[0], numResults);
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
