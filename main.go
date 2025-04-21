package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/philippgille/chromem-go"
	"github.com/sashabaranov/go-openai"
)

type MemoryServer struct {
	db       *chromem.DB
	aiClient *openai.Client
}

func NewMemoryServer(dbPath string, openAIKey string) (*MemoryServer, error) {
	// Create or open the database
	db, err := chromem.NewPersistentDB(dbPath, true)
	if err != nil {
		return nil, fmt.Errorf("failed to create/open database: %w", err)
	}

	// Create OpenAI client for embeddings
	client := openai.NewClient(openAIKey)

	return &MemoryServer{
		db:       db,
		aiClient: client,
	}, nil
}

func (ms *MemoryServer) generateEmbedding(text string) ([]float32, error) {
	queryReq := openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.AdaEmbeddingV2,
	}

	queryResponse, err := ms.aiClient.CreateEmbeddings(context.Background(), queryReq)
	if err != nil {
		return nil, fmt.Errorf("error creating embedding: %w", err)
	}

	// Convert float64 to float32
	embedding := queryResponse.Data[0].Embedding
	result := make([]float32, len(embedding))
	for i, v := range embedding {
		result[i] = float32(v)
	}

	return result, nil
}

func main() {
	// Get environment variables
	dbPath := os.Getenv("MEMORY_DB_PATH")
	if dbPath == "" {
		dbPath = "./memory"
	}

	openAIKey := os.Getenv("OPENAI_API_KEY")
	if openAIKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable required")
	}

	// Create memory server
	memServer, err := NewMemoryServer(dbPath, openAIKey)
	if err != nil {
		log.Fatalf("Failed to create memory server: %v", err)
	}

	// Create MCP server
	s := server.NewMCPServer(
		"Efficient Memory Server",
		"1.0.0",
		server.WithResourceCapabilities(true, true),
	)

	// Get or create collection
	collection, err := memServer.db.GetOrCreateCollection("memories", nil, nil)
	if err != nil {
		log.Fatalf("Failed to get/create collection: %v", err)
	}

	// Add memory storage tool
	addMemoryTool := mcp.NewTool("add_memory",
		mcp.WithDescription("Add a new memory with semantic search capabilities"),
		mcp.WithString("content",
			mcp.Required(),
			mcp.Description("The content to remember"),
		),
		mcp.WithString("metadata",
			mcp.Description("Additional metadata as JSON string"),
		),
	)

	s.AddTool(addMemoryTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		content, ok := request.Params.Arguments["content"].(string)
		if !ok {
			return mcp.NewToolResultError("content must be a string"), nil
		}

		metadata := ""
		if m, ok := request.Params.Arguments["metadata"].(string); ok {
			metadata = m
		}

		// Generate embedding
		embedding, err := memServer.generateEmbedding(content)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to generate embedding: %v", err)), nil
		}

		// Create document
		doc := chromem.Document{
			ID:        fmt.Sprintf("mem_%d", time.Now().UnixNano()),
			Metadata:  map[string]string{"raw_metadata": metadata},
			Embedding: embedding,
			Content:   content,
		}

		// Add to collection
		err = collection.AddDocument(ctx, doc)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to add document: %v", err)), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Memory stored with ID: %s", doc.ID)), nil
	})

	// Add semantic search tool
	searchTool := mcp.NewTool("search_memory",
		mcp.WithDescription("Search for memories by semantic similarity"),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("Search query text"),
		),
		mcp.WithNumber("limit",
			mcp.Description("Maximum number of results (default: 5)"),
			mcp.Min(1),
			mcp.Max(20),
		),
	)

	s.AddTool(searchTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, ok := request.Params.Arguments["query"].(string)
		if !ok {
			return mcp.NewToolResultError("query must be a string"), nil
		}

		limit := 5
		if l, ok := request.Params.Arguments["limit"].(float64); ok {
			limit = int(l)
		}

		// Generate query embedding
		queryEmbedding, err := memServer.generateEmbedding(query)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to generate query embedding: %v", err)), nil
		}

		// Search for similar documents
		results, err := collection.QueryEmbedding(ctx, queryEmbedding, limit, nil, nil)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("search failed: %v", err)), nil
		}

		if len(results) == 0 {
			return mcp.NewToolResultText("No matching memories found."), nil
		}

		// Format results
		response := ""
		for i, result := range results {
			response += fmt.Sprintf("[%d] %s (similarity: %.3f)\n",
				i+1, result.Content, result.Similarity)
			if metadata, ok := result.Metadata["raw_metadata"]; ok && metadata != "" {
				response += fmt.Sprintf("   Metadata: %s\n", metadata)
			}
		}

		return mcp.NewToolResultText(response), nil
	})

	// Add resource for collection stats
	statsResource := mcp.NewResource(
		"memory://stats",
		"Memory Statistics",
		mcp.WithResourceDescription("Statistics about stored memories"),
		mcp.WithMIMEType("application/json"),
	)

	s.AddResource(statsResource, func(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
		count := collection.Count()

		stats := fmt.Sprintf(`{"total_memories": %d}`, count)

		return []mcp.ResourceContents{
			mcp.TextResourceContents{
				URI:      "memory://stats",
				MIMEType: "application/json",
				Text:     stats,
			},
		}, nil
	})

	// Start the server
	if err := server.ServeStdio(s); err != nil {
		fmt.Printf("Server error: %v\n", err)
	}
}
