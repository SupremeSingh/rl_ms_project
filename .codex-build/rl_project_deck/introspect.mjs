import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/manmit/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx'));
console.log('slides keys', Object.keys(p.slides));
console.log('proto slides keys', Object.keys(p.toProto().slides ?? {}));
console.log('proto root keys', Object.keys(p.toProto()));
console.log('slide methods', Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides)));
console.log('first slide methods', Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides.getItem(0))));
