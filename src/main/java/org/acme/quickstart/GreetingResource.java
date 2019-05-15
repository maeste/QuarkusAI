package org.acme.quickstart;

import org.jboss.resteasy.plugins.providers.multipart.InputPart;
import org.jboss.resteasy.plugins.providers.multipart.MultipartFormDataInput;

import javax.inject.Inject;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.MultivaluedMap;
import javax.ws.rs.core.Response;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

@Path("/hello")
public class GreetingResource {

    @Inject
    GreetingService service;

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String hello() {
        return "hello";
    }

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    @Path("/greeting/{name}")
    public String greeting(@PathParam("name") String name) {
        return service.greeting(name);
    }
    @POST
    @Path("/loadImage")
    @Consumes("multipart/form-data")
    public Response loadImage(MultipartFormDataInput input) throws Exception {
        Map<String, List<InputPart>> formParts = input.getFormDataMap();
        File f = storeWalletFile(formParts.get("file").iterator().next());
        //String password = formParts.get("password").iterator().next().getBodyAsString();
        byte[] bytes = Files.readAllBytes(Paths.get(f.getAbsolutePath()));
        String returnString = LabelImage.labelImage(f.getName(), bytes);
        return Response.status(200).entity("labeled image = " + returnString).build();
    }

    private File storeWalletFile(InputPart inputPart) throws IOException {
        MultivaluedMap<String, String> headers = inputPart.getHeaders();
        String fileName = parseFileName(headers);
        InputStream istream = inputPart.getBody(InputStream.class, null);
        File f = new File(System.getProperty("java.io.tmpdir"), fileName);
        saveFile(istream, f);
        return f;
    }

    // Parse Content-Disposition header to get the original file name
    private String parseFileName(MultivaluedMap<String, String> headers) {
        String[] contentDispositionHeader = headers.getFirst("Content-Disposition").split(";");
        for (String name : contentDispositionHeader) {
            if ((name.trim().startsWith("filename"))) {
                String[] tmp = name.split("=");
                String fileName = tmp[1].trim().replaceAll("\"", "");
                return fileName;
            }
        }
        return "randomName";
    }

    private void saveFile(InputStream uploadedInputStream, File serverLocation) {
        try {
            OutputStream outpuStream = new FileOutputStream(serverLocation);
            int read = 0;
            byte[] bytes = new byte[1024];
            while ((read = uploadedInputStream.read(bytes)) != -1) {
                outpuStream.write(bytes, 0, read);
            }
            outpuStream.flush();
            outpuStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
