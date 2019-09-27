package org.acme.quickstart;

import java.util.Objects;

public final class Probability
{
   private String label;
   private float percentage;
   
   public Probability() {
      
   }

   public Probability(String label, float percentage)
   {
      super();
      this.label = label;
      this.percentage = percentage;
   }

   public String getLabel()
   {
      return label;
   }

   public void setLabel(String label)
   {
      this.label = label;
   }

   public float getPercentage()
   {
      return percentage;
   }

   public void setPercentage(float percentage)
   {
      this.percentage = percentage;
   }
   
   
   @Override
   public boolean equals(Object obj) {
       if (!(obj instanceof Probability)) {
           return false;
       }

       Probability other = (Probability) obj;

       return Objects.equals(other.label, this.label) && Objects.equals(other.percentage, this.percentage);
   }

   @Override
   public int hashCode() {
       return Objects.hash(this.label, this.percentage);
   }
   
}
