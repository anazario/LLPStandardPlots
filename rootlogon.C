{
   // ROOT logon script to register custom colors for unrolled plotting
   // This ensures colors persist across ROOT sessions
   
   printf("Loading custom colors for unrolled plotting...\n");
   
   // Register the custom hex colors used in sv_variables_comparison.py
   vector<TString> hex_colors = {
      "#5A4484",  // Purple
      "#347889",  // Teal  
      "#F4B240",  // Orange
      "#E54B26",  // Red
      "#C05780",  // Pink
      "#7A68A6",  // Blue
      "#2E8B57",  // Green
      "#8B4513"   // Brown
   };
   
   // Create and register each color with explicit indices
   vector<Int_t> color_indices;
   for (int i = 0; i < hex_colors.size(); i++) {
      Int_t color_index = TColor::GetColor(hex_colors[i]);
      color_indices.push_back(color_index);
      
      // Also create a named color for easier reference
      TColor* color = gROOT->GetColor(color_index);
      if (color) {
         printf("  Registered color %s as index %d (R=%.0f, G=%.0f, B=%.0f)\n", 
                hex_colors[i].Data(), color_index, 
                color->GetRed()*255, color->GetGreen()*255, color->GetBlue()*255);
      } else {
         printf("  Created color %s as index %d\n", hex_colors[i].Data(), color_index);
      }
   }
   
   printf("Custom colors loaded successfully!\n");
   printf("Color indices: ");
   for (int i = 0; i < color_indices.size(); i++) {
      printf("%d ", color_indices[i]);
   }
   printf("\n");
   
   // Create transparent orange color for postfit plots
   Int_t transparent_orange = TColor::GetColorTransparent(kOrange+7, 0.7);
   printf("Registered transparent orange as index %d\n", transparent_orange);
   
   printf("\nTo restore colors when opening ROOT files:\n");
   printf("  1. Run this rootlogon.C or\n");
   printf("  2. In the ROOT file, look for 'summary/restore_colors_macro' and execute it\n");
}