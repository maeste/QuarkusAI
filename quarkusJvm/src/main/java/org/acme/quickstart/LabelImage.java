package org.acme.quickstart;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import com.google.common.io.ByteStreams;
import org.tensorflow.types.UInt8;

import javax.imageio.ImageIO;

public final class LabelImage
{
   private static List<String> labels = loadLabels();
   private static byte[] graphDef = loadBytes("mobilenet_frozen.pb");

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

   public static List<Probability> labelImage(String fileName, InputStream is) throws Exception
   {
      initSession();

      float[][] probabilities = null;
      try (Tensor<Float> input = makeImageTensor(is);Tensor<Float> output = feedAndRun(s, input))
      {
         probabilities = extractProbabilities(output);
         List<Probability> result = new ArrayList<>(labels.size());
         for (int i = 0; i < labels.size(); i++) {
            result.add(new Probability(labels.get(i), probabilities[0][i]));
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

   private static Tensor<Float> makeImageTensor(InputStream is) throws IOException {
      long millis = System.currentTimeMillis();BufferedImage img = ImageIO.read(is);
      //if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
         BufferedImage newImage = new BufferedImage(
                 256, 256, BufferedImage.TYPE_3BYTE_BGR);
         Graphics2D g = newImage.createGraphics();
         g.drawImage(img, 0, 0, 256, 256, null);
         g.dispose();
         img = newImage;
      //}

      byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
      // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
      bgr2rgb(data);
      final long BATCH_SIZE = 1;
      final long CHANNELS = 3;
      long[] shape = new long[] {BATCH_SIZE, 128, 128, CHANNELS};
      System.out.println(System.currentTimeMillis() - millis);
      return Tensor.create(Float.class, shape, ByteBuffer.wrap(data));
   }

   private static void bgr2rgb(byte[] data) {
      for (int i = 0; i < data.length; i += 3) {
         byte tmp = data[i];
         data[i] = data[i + 2];
         data[i + 2] = tmp;
      }
   }
   private static Tensor<Float> feedAndRun(Session session, Tensor<Float> input)
   {
      return session.runner().feed("input", input).fetch("MobilenetV1/Predictions/Reshape_1").run().get(0)
            .expect(Float.class);
   }

   private static float[][] extractProbabilities(Tensor<Float> output)
   {
      float[][] probabilities = new float[(int) output.shape()[0]][(int) output.shape()[1]];
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
