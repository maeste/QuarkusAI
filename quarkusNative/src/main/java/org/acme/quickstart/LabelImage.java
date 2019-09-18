package org.acme.quickstart;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.google.common.io.ByteStreams;

public final class LabelImage
{
   private static List<String> labels = loadLabels();
   private static volatile boolean reloaded = false;
   private static byte[] graphDef = warmUpAndDumpGraphDef();

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

   public static String labelImage(String fileName, byte[] bytes) throws Exception
   {
      graalVmHack();
      initSession();
      float[] probabilities = null;
      try (Tensor<String> input = Tensors.create(bytes); Tensor<Float> output = feedAndRun(s, input))
      {
         probabilities = extractProbabilities(output);
         int label = argmax(probabilities);
         return String.format("%-30s --> %-15s (%.2f%% likely)\n", fileName, labels.get(label),
               probabilities[label] * 100.0);
      }
   }

   private static void graalVmHack() throws Exception
   {
      if (reloaded)
         return;
      Method tfInit = Class.forName("org.tensorflow.TensorFlow").getDeclaredMethod("init");
      tfInit.setAccessible(true);
      tfInit.invoke(null);
      reloaded = true;
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

   private static byte[] warmUpAndDumpGraphDef()
   {
      Graph graph = new Graph();
      graph.importGraphDef(loadBytes("graph.pb"));
      try (Session ss = new Session(graph);
            Tensor<String> input = Tensors.create(loadBytes("bg.png"));
            Tensor<Float> output = feedAndRun(ss, input))
      {
         float[] probabilities = extractProbabilities(output);
         int label = argmax(probabilities);
         System.out.println(
               String.format("Warm-up: %-15s (%.2f%% likely)", labels.get(label), probabilities[label] * 100.0));
         return graph.toGraphDef();
      }
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

   private static int argmax(float[] probabilities)
   {
      int best = 0;
      for (int i = 1; i < probabilities.length; ++i)
      {
         if (probabilities[i] > probabilities[best])
         {
            best = i;
         }
      }
      return best;
   }

}
