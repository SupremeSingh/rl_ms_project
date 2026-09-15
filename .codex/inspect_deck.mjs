import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePath = "/Users/manmit/Downloads/math_rl_project_briefing.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
  maxChars: 30000,
});
console.log(snapshot.ndjson);
console.log("SLIDE_COUNT", presentation.slides.items.length);
console.log("SIZE", JSON.stringify(presentation.slideSize));
