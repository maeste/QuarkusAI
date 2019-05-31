package org.acme.quickstart;/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import com.google.common.io.ByteStreams;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Simplified version of
 * https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
 */
public class LabelImageStatic {

  private static Graph staticGraph;
  private static Session staticSession;
  private static List<String> labels;
  static {
    try {
      staticGraph = new Graph();
      staticSession = new Session(staticGraph);
      staticGraph.importGraphDef(loadGraphDef());
      labels = loadLabels();

    } catch (Exception e) {

    }
  }

  public String labelImageStatic(String fileName, byte[] bytes) throws Exception {
      float[] probabilities = null;
      try (Tensor<String> input = Tensors.create(bytes);
           Tensor<Float> output =
                   staticSession
                           .runner()
                           .feed("encoded_image_bytes", input)
                           .fetch("probabilities")
                           .run()
                           .get(0)
                           .expect(Float.class)) {
        if (probabilities == null) {
          probabilities = new float[(int) output.shape()[0]];
        }
        output.copyTo(probabilities);
        int label = argmax(probabilities);
        return String.format("%-30s --> %-15s (%.2f%% likely)\n",
                fileName, labels.get(label), probabilities[label] * 100.0);
      }


  }

  private static byte[] loadGraphDef() throws IOException {
    try (InputStream is = LabelImageStatic.class.getClassLoader().getResourceAsStream("graph.pb")) {
      return ByteStreams.toByteArray(is);
    }
  }

  private static ArrayList<String> loadLabels() throws IOException {
    ArrayList<String> labels = new ArrayList<String>();
    String line;
    final InputStream is = LabelImageStatic.class.getClassLoader().getResourceAsStream("labels.txt");
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
      while ((line = reader.readLine()) != null) {
        labels.add(line);
      }
    }
    return labels;
  }

  private int argmax(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }





}
