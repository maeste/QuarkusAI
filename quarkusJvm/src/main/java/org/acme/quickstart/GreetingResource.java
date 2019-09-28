package org.acme.quickstart;

import java.io.InputStream;
import java.util.List;

import javax.ws.rs.Consumes;
import javax.ws.rs.HeaderParam;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;

import org.jboss.resteasy.plugins.providers.multipart.InputPart;
import org.jboss.resteasy.plugins.providers.multipart.MultipartFormDataInput;

@Path("/quarkusai")
public class GreetingResource {

    @POST
    @Path("/labelImageJvm/{results}")
    @Consumes("multipart/form-data")
    @Produces("application/json")
    public ImageProcessingResult loadImage(@HeaderParam("Content-Length") String contentLength, @PathParam("results") int results, MultipartFormDataInput input) throws Exception {
        long before = System.currentTimeMillis();
        InputPart inputPart = input.getFormDataMap().get("file").iterator().next();
        List<Probability> probs = ImageProcessor.labelImage(inputPart.getBody(InputStream.class, null), results);
        return new ImageProcessingResult((System.currentTimeMillis() - before), probs);
    }
}
