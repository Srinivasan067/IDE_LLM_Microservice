const fs = require('fs');
const pdf = require('pdf-parse');
const { MongoClient } = require('mongodb');
const OpenAI = require('openai');
require('dotenv').config();

const uri = process.env.Mongo_Url;
const client = new MongoClient(uri);
const openai = new OpenAI({ apiKey: process.env.GITHUB_TOKEN, baseURL: "https://models.github.ai/inference/" });

async function embedText(text) {
  const response = await openai.embeddings.create({
    input: text,
    model: "openai/text-embedding-3-small"
  });
  return response.data[0].embedding;
}

async function main() {
  const dataBuffer = fs.readFileSync('RAG_Car_Availability_Dataset_with_RAG_Examples.pdf');
  const data = await pdf(dataBuffer);
  const text = data.text;

  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  const chunks = [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('Query:')) {
      // Combine Query and next Response
      const qa = lines[i] + ' ' + (lines[i + 1] || '');
      if (qa.length >= 20) chunks.push(qa);
      i++; // skip next line (the Response)
    } else if (
      !lines[i].startsWith('Response:') &&
      lines[i].length >= 20
    ) {
      chunks.push(lines[i]);
    }
  }

  await client.connect();
  const db = client.db();
  const collection = db.collection('vectors');

  for (const chunk of chunks) {
    if (chunk.trim().length < 20) continue; // skip very short chunks
    const embedding = await embedText(chunk);
    await collection.insertOne({ chunk, embedding });
    console.log('Inserted chunk:', chunk.slice(0, 60));
  }

  await client.close();
  console.log('PDF ingestion complete!');
}

main().catch(console.error);