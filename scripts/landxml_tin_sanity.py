from __future__ import annotations

from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from raster_compare.landxml_tin import (
    resolve_landxml_tin_config,
    run_landxml_tin_to_mesh,
)


LANDXML_MINIMAL = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<LandXML xmlns=\"http://www.landxml.org/schema/LandXML-1.2\">
  <Surfaces>
    <Surface name=\"TIN_TEST\">
      <Definition surfType=\"TIN\">
        <Pnts>
          <P id=\"1\">0 0 0</P>
          <P id=\"2\">1 0 0</P>
          <P id=\"3\">0 1 0</P>
        </Pnts>
        <Faces>
          <F>1 2 3</F>
        </Faces>
      </Definition>
    </Surface>
  </Surfaces>
</LandXML>
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        xml_path = tmpdir_path / "minimal.xml"
        xml_path.write_text(LANDXML_MINIMAL, encoding="utf-8")

        raw_config = {
            "pipeline": "landxml_tin_to_mesh",
            "landxml_tin_to_mesh": {
                "name": "sanity_landxml",
                "outdir": str(tmpdir_path / "outputs"),
                "excel": True,
                "input_xml": str(xml_path),
                "surface_name": "TIN_TEST",
            },
        }

        config = resolve_landxml_tin_config(raw_config)
        outputs = run_landxml_tin_to_mesh(config)

        obj_path = Path(outputs["obj"])
        vertices_csv = Path(outputs["vertices_csv"])
        faces_csv = Path(outputs["faces_csv"])

        assert obj_path.exists()
        assert vertices_csv.exists()
        assert faces_csv.exists()

        obj_text = obj_path.read_text(encoding="utf-8")
        assert obj_text.count("\nv ") == 3
        assert obj_text.count("\nf ") == 1

        print("landxml_tin_sanity ok")


if __name__ == "__main__":
    main()
