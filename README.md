## Qdrant on fly.io

in fly.toml set an ENV KEY, in a production application you should use fly secrets

```
[env]
  QDRANT__SERVICE__API_KEY = "MY_SUPER_SECRET_KEY"
```

```
import { QdrantClient } from "@qdrant/js-client-rest";

const qdrant = new QdrantClient({
  host: "<your_appname.fly.dev",
  port: 443,
  apiKey: "MY_SUPER_SECRET_KEY",
  checkCompatibility: false,
});
```

### Sample beta testing abstraction

```
import { QdrantClient } from "@qdrant/js-client-rest";
import { createHash } from "crypto";
import { OpenAI } from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const qdrant = new QdrantClient({
  host: "qdrant-db.fly.dev",
  port: 443,
  apiKey: "MY_SUPER_SECRET_KEY",
  checkCompatibility: false,
});

// Constants for embedding models
const NLP_MODEL = "text-embedding-3-small";
const CODE_MODEL = "text-embedding-ada-002";

// Generate UUID from file path
function generateUuidFromFilePath(path) {
  return createHash("md5").update(path).digest("hex");
}

// Parse content into chunks (simplified version)
async function parseFileIntoChunks({ path, content }) {
  const chunkSize = 1000;
  const chunks = [];

  for (let i = 0; i < content.length; i += chunkSize) {
    const chunkContent = content.slice(i, i + chunkSize);
    const chunkId = generateUuidFromFilePath(`${path}#chunk-${i}`);

    chunks.push({
      id: chunkId,
      path,
      content: chunkContent,
      contentHash: createHash("md5").update(chunkContent).digest("hex"),
      loc: { start: i, end: Math.min(i + chunkSize, content.length) },
      metadata: {},
    });
  }

  return chunks;
}

//
// Core API - Refactored to use collectionName instead of projectId for reusability
//

export default class Vector {
  constructor(collectionName) {
    this.collectionName = collectionName;
  }

  async vectorFile(file) {
    const embedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: file.content,
    });

    return embedding.data[0].embedding;
  }

  //
  //
  //
  async getCollection(collectionName) {
    try {
      const collectionInfo = await qdrant.getCollection(collectionName);
      return collectionInfo;
    } catch (error) {
      throw Error(`Collection ${collectionName} not found`);
    }
  }

  //
  // Create a collection in the vector DB
  //
  async createCollection() {
    // console.log("qdrant", qdrant);
    let collection;
    const collectionName = this.collectionName;

    // 1. Create collection with multi-vector support (if not exists)
    try {
      collection = await qdrant.createCollection(collectionName, {
        vectors: {
          nlp: {
            size: 1536, // text-embedding-3-small dimension
            distance: "Cosine",
          },
          code: {
            size: 1536, // text-embedding-ada-002 dimension (fallback)
            distance: "Cosine",
          },
        },
      });
      console.log("Multi-vector collection created successfully");
    } catch (createError) {
      // Collection might already exist, that's okay
      console.log("Collection creation info:", createError.message);
    }

    return collection;
  }

  //
  // Ensure a collection exists in the vector DB
  //
  async ensureCollection() {
    let collection;
    try {
      collection = await qdrant.getCollection(this.collectionName);
    } catch (err) {
      // If collection does not exist, create it
      console.log(`Collection ${this.collectionName} missing. Creating...`);
      await qdrant.createCollection(this.collectionName, {
        vectors: {
          nlp: { size: 1536, distance: "Cosine" },
          code: { size: 1536, distance: "Cosine" },
        },
      });

      // After creation, fetch the collection again to return its info
      collection = await qdrant.getCollection(this.collectionName);
    }
    return collection;
  }

  async delete(path) {
    const pointId = generateUuidFromFilePath(path);
    await qdrant.delete(this.collectionName, {
      points: [pointId],
    });
  }

  async deleteCollection() {
    const response = await qdrant.deleteCollection(this.collectionName);
    console.log("Collection deleted", response);
    return response;
  }

  async upsert(path, content) {
    // Parse the file content into chunks
    const chunks = await parseFileIntoChunks({ path, content });

    // Generate embeddings for all chunks
    const embeddedPoints = await Promise.all(chunks.map((chunk, index) => this.embed(chunk, index)));

    // Upsert all chunks (this will update if exists, insert if not)
    const result = await qdrant.upsert(this.collectionName, {
      points: embeddedPoints,
    });

    return result;
  }

  async create() {
    await this.ensureCollection();
  }

  async embed(chunk, index) {
    // Generate dual embeddings for hybrid search
    const [nlpEmbedding, codeEmbedding] = await Promise.all([
      openai.embeddings.create({
        model: NLP_MODEL,
        input: chunk.content,
      }),
      openai.embeddings.create({
        model: CODE_MODEL,
        input: chunk.content,
      }),
    ]);

    return {
      id: chunk.id,
      vector: {
        nlp: nlpEmbedding.data[0].embedding,
        code: codeEmbedding.data[0].embedding,
      },
      payload: {
        collectionName: this.collectionName,
        path: chunk.path,
        chunkType: chunk.chunkType,
        name: chunk.name,
        contentLength: chunk.content.length,
        content: chunk.content,
        contentHash: chunk.contentHash,
        loc: chunk.loc,
        metadata: chunk.metadata || {},
      },
    };
  }

  async search(query, limit = 5, chunkType = null) {
    const collection = await this.ensureCollection();

    // Generate embeddings for both vector spaces
    const [nlpQueryEmbedding, codeQueryEmbedding] = await Promise.all([
      openai.embeddings.create({
        model: NLP_MODEL,
        input: query,
      }),
      openai.embeddings.create({
        model: CODE_MODEL,
        input: query,
      }),
    ]);

    // Prepare search filters
    const searchFilter = chunkType
      ? {
          must: [
            {
              key: "chunkType",
              match: { value: chunkType },
            },
          ],
        }
      : undefined;

    // Search both vector spaces in parallel
    const [nlpResults, codeResults] = await Promise.all([
      qdrant.search(this.collectionName, {
        vector: { name: "nlp", vector: nlpQueryEmbedding.data[0].embedding },
        limit: Math.ceil(limit * 1.5), // Get more results for merging
        with_payload: true,
        filter: searchFilter,
      }),
      qdrant.search(this.collectionName, {
        vector: { name: "code", vector: codeQueryEmbedding.data[0].embedding },
        limit: Math.ceil(limit * 1.5),
        with_payload: true,
        filter: searchFilter,
      }),
    ]);

    // Merge and deduplicate results, preferring NLP for semantic matches
    const resultMap = new Map();
    const mergedResults = [];

    // Add NLP results first (higher priority for semantic understanding)
    nlpResults.forEach((result) => {
      const key = result.id;
      if (!resultMap.has(key)) {
        resultMap.set(key, {
          ...result,
          searchType: "nlp",
          nlpScore: result.score,
          codeScore: 0,
        });
      }
    });

    // Add code results, updating scores if already present
    codeResults.forEach((result) => {
      const key = result.id;
      if (resultMap.has(key)) {
        // Update existing result with code score
        const existing = resultMap.get(key);
        existing.codeScore = result.score;
        existing.combinedScore = existing.nlpScore * 0.6 + result.score * 0.4; // Weight NLP higher
      } else {
        // Add new code-only result
        resultMap.set(key, {
          ...result,
          searchType: "code",
          nlpScore: 0,
          codeScore: result.score,
          combinedScore: result.score * 0.4,
        });
      }
    });

    // Convert to array and sort by combined score
    mergedResults.push(...resultMap.values());
    mergedResults.sort((a, b) => {
      // Prefer results that appear in both searches
      const aHasBoth = a.nlpScore > 0 && a.codeScore > 0;
      const bHasBoth = b.nlpScore > 0 && b.codeScore > 0;

      if (aHasBoth && !bHasBoth) return -1;
      if (!aHasBoth && bHasBoth) return 1;

      // Then sort by combined score
      return (b.combinedScore || b.score) - (a.combinedScore || a.score);
    });

    // Group by module to diversify results (optional)
    const diversifiedResults = this.diversifyByModule(mergedResults, limit);

    return diversifiedResults.slice(0, limit);
  }

  // Helper method to diversify results by module
  diversifyByModule(results, limit) {
    const moduleGroups = new Map();
    const diversified = [];

    // Group by module

    results.forEach((result) => {
      const moduleName = result.payload.metadata?.module || "root";
      if (!moduleGroups.has(moduleName)) {
        moduleGroups.set(moduleName, []);
      }
      moduleGroups.get(moduleName).push(result);
    });

    // Round-robin selection from different modules
    const moduleIterators = Array.from(moduleGroups.values()).map((group) => ({ group, index: 0 }));
    let currentModule = 0;

    while (diversified.length < limit && moduleIterators.some((iter) => iter.index < iter.group.length)) {
      const iterator = moduleIterators[currentModule];
      if (iterator.index < iterator.group.length) {
        diversified.push(iterator.group[iterator.index]);
        iterator.index++;
      }
      currentModule = (currentModule + 1) % moduleIterators.length;
    }

    return diversified;
  }
}

```
