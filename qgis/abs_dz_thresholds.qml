<qgis version="3.28" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1">
      <rastershader>
        <colorrampshader colorRampType="DISCRETE" classificationMode="EqualInterval" clip="0">
          <item alpha="255" value="0.10" label="<= 0.10" color="247,252,245,255"/>
          <item alpha="255" value="0.25" label="0.10 - 0.25" color="199,233,192,255"/>
          <item alpha="255" value="0.50" label="0.25 - 0.50" color="116,196,118,255"/>
          <item alpha="255" value="1.00" label="0.50 - 1.00" color="35,139,69,255"/>
          <item alpha="255" value="9999" label="> 1.00" color="0,90,50,255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
