// This file will run ONLY on Bun.js
// to run in node replace Bun.file.text with fs.readFileSync,utf-8
import { readdirSync } from "fs";

const files = readdirSync("./past"); // Get all files in the 'past' directory

// EXTRACT ALL PAPERS
const globalPapers: Array<string> = [];
for await (const file of files) {
  if (!file.endsWith(".md")) continue; // Skip non-markdown files
  const text = await Bun.file(`./past/${file}`).text(); // Read the file
  const papers = text.match(/<pa-per\b[^>]*>[\s\S]*?<\/pa-per>/gi) || []; // Find all <pa-per> tags

  globalPapers.push(...papers); // Add them to the global array
};


// EXTRACT ALL METADATA
const attriMatch = /(\S+)=["']?((?:.(?!["']?\s+(?:\S+)=|[>"']))+.)["']?/g; // Regex to match attributes

const globalPapersObj: Array<Map<string, string>> = []; // Array of objects
for await (const paper of globalPapers) {
  const paperMap: Map<string, string> = new Map(); // Object to store attributes
  const attrs = paper.match(attriMatch) || []; // Get all attributes

  for await (const attr of attrs) {
    const [key, value] = attr.split("="); // Split the attribute into key and value
    paperMap.set(key, value.replaceAll("\"", "")); // Add the attribute to the object
  };

  globalPapersObj.push(paperMap); // Add the object to the array
};

// INDIVIDUAL FIXES
for await (const paper of globalPapersObj) {
  const href = paper.get("href"); // Get the href attribute
  if (href.startsWith("https://")) continue; // Skip if it's a link
  paper.set("href", `https://${href}`); // Add https
};

console.log(globalPapersObj);
