package org.acme.quickstart;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;

import org.nd4j.shade.jackson.databind.ObjectMapper;

public class ImageNetLabels {

    //FIXME
    private final static String jsonUrl =
                    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json";
    private static ArrayList<String> predictionLabels = getLabels();

    @SuppressWarnings("unchecked")
	public static ArrayList<String> getLabels() {
        if (predictionLabels == null) {
            HashMap<String, ArrayList<String>> jsonMap;
            try {
                jsonMap = new ObjectMapper().readValue(new URL(jsonUrl), HashMap.class);
                predictionLabels = new ArrayList<>(jsonMap.size());
                for (int i = 0; i < jsonMap.size(); i++) {
                    predictionLabels.add(jsonMap.get(String.valueOf(i)).get(1));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return predictionLabels;
    }
}
