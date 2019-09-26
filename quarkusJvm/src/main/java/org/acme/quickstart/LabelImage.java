package org.acme.quickstart;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.google.common.io.ByteStreams;

public final class LabelImage
{
   private static List<String> labels = loadLabels();
   private static byte[] graphDef = loadBytes("graph.pb");

   private static volatile Session s;

   private static void initSession()
   {
      if (s == null)
      {
         synchronized (LabelImage.class)
         {
            if (s == null)
            {
               Graph graph = new Graph();
               graph.importGraphDef(graphDef);
               s = new Session(graph);
            }
         }
      }
   }

   public static List<Probability> labelImage(String fileName, byte[] bytes) throws Exception
   {
      initSession();
      float[] probabilities = null;
      try (Tensor<String> input = Tensors.create(bytes); Tensor<Float> output = feedAndRun(s, input))
      {
         probabilities = extractProbabilities(output);
         List<Probability> result = new ArrayList<>(labels.size());
         for (int i = 0; i < labels.size(); i++) {
            result.add(new Probability(labels.get(i), probabilities[i]));
         }
		 result.sort(new Comparator<Probability>() {
			@Override
			public int compare(Probability o1, Probability o2) {
				return Float.compare(o2.getPercentage(), o1.getPercentage());
			}
		 });
         return result;
      }
   }

   private static Tensor<Float> feedAndRun(Session session, Tensor<String> input)
   {
      return session.runner().feed("encoded_image_bytes", input).fetch("probabilities").run().get(0)
            .expect(Float.class);
   }

   private static float[] extractProbabilities(Tensor<Float> output)
   {
      float[] probabilities = new float[(int) output.shape()[0]];
      output.copyTo(probabilities);
      return probabilities;
   }

   private static byte[] loadBytes(String resource)
   {
      System.out.println("Load bytes: " + resource);
      try (InputStream is = LabelImage.class.getClassLoader().getResourceAsStream(resource))
      {
         return ByteStreams.toByteArray(is);
      }
      catch (Exception e)
      {
         throw new RuntimeException(e);
      }
   }

   private static ArrayList<String> loadLabels()
   {
      try
      {
         System.out.println("Load labels!");
         ArrayList<String> labels = new ArrayList<String>();
         String line;
         final InputStream is = LabelImage.class.getClassLoader().getResourceAsStream("labels.txt");
         try (BufferedReader reader = new BufferedReader(new InputStreamReader(is)))
         {
            while ((line = reader.readLine()) != null)
            {
               labels.add(line);
            }
         }
         return labels;
      }
      catch (Exception e)
      {
         throw new RuntimeException(e);
      }
   }
}
