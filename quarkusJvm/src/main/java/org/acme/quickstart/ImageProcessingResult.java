package org.acme.quickstart;

import java.util.List;

import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public final class ImageProcessingResult
{
   private long processingTime;
   private List<Probability> probabilities;
   
   public ImageProcessingResult(long processingTime, List<Probability> probabilities)
   {
      super();
      this.processingTime = processingTime;
      this.probabilities = probabilities;
   }
   public ImageProcessingResult()
   {
      super();
   }
   public long getProcessingTime()
   {
      return processingTime;
   }
   public void setProcessingTime(long processingTime)
   {
      this.processingTime = processingTime;
   }
   public List<Probability> getProbabilities()
   {
      return probabilities;
   }
   public void setProbabilities(List<Probability> probabilities)
   {
      this.probabilities = probabilities;
   }
}
