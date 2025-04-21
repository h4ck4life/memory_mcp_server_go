package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/search/query"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	bolt "go.etcd.io/bbolt"
)

// Memory represents a stored memory item
type Memory struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Type      string    `json:"type"`
	Tags      []string  `json:"tags"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// MemoryStore manages memory with BoltDB and Bleve search
type MemoryStore struct {
	db    *bolt.DB
	index bleve.Index
}

const (
	memoryBucket = "memories"
)

// NewMemoryStore creates a new memory store
func NewMemoryStore(dbPath string) (*MemoryStore, error) {
	// Open BoltDB
	db, err := bolt.Open(dbPath, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Create bucket if it doesn't exist
	err = db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(memoryBucket))
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bucket: %w", err)
	}

	// Create or open Bleve index
	indexPath := dbPath + ".bleve"
	var index bleve.Index

	// Check if index exists
	if idx, err := bleve.Open(indexPath); err == nil {
		index = idx
	} else {
		// Create new index
		mapping := bleve.NewIndexMapping()
		var createErr error
		index, createErr = bleve.New(indexPath, mapping)
		if createErr != nil {
			return nil, fmt.Errorf("failed to create index: %w", createErr)
		}

		// Index existing memories
		err = db.View(func(tx *bolt.Tx) error {
			b := tx.Bucket([]byte(memoryBucket))
			return b.ForEach(func(k, v []byte) error {
				var memory Memory
				if err := json.Unmarshal(v, &memory); err != nil {
					return err
				}
				return index.Index(memory.ID, memory)
			})
		})
		if err != nil {
			return nil, fmt.Errorf("failed to index existing memories: %w", err)
		}
	}

	return &MemoryStore{
		db:    db,
		index: index,
	}, nil
}

// Close closes the database and index
func (ms *MemoryStore) Close() error {
	if err := ms.index.Close(); err != nil {
		return err
	}
	return ms.db.Close()
}

// Add adds a new memory
func (ms *MemoryStore) Add(content, memType string, tags []string) (string, error) {
	memory := Memory{
		ID:        fmt.Sprintf("mem_%d", time.Now().UnixNano()),
		Content:   content,
		Type:      memType,
		Tags:      tags,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Save to BoltDB
	err := ms.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(memoryBucket))
		data, err := json.Marshal(memory)
		if err != nil {
			return err
		}
		return b.Put([]byte(memory.ID), data)
	})
	if err != nil {
		return "", fmt.Errorf("failed to save memory: %w", err)
	}

	// Index for search
	if err := ms.index.Index(memory.ID, memory); err != nil {
		return "", fmt.Errorf("failed to index memory: %w", err)
	}

	return memory.ID, nil
}

// Search searches for memories by text and/or tags
func (ms *MemoryStore) Search(queryString string, tags []string) ([]Memory, error) {
	// Build search query
	var searchQuery query.Query
	var queries []query.Query

	// Text search if query is provided
	if queryString != "" {
		matchQuery := bleve.NewMatchQuery(queryString)
		matchQuery.SetField("Content")
		queries = append(queries, matchQuery)
	}

	// Tag search
	for _, tag := range tags {
		termQuery := bleve.NewTermQuery(tag)
		termQuery.SetField("Tags")
		queries = append(queries, termQuery)
	}

	// Combine queries
	if len(queries) > 0 {
		if len(queries) == 1 {
			searchQuery = queries[0]
		} else {
			searchQuery = bleve.NewConjunctionQuery(queries...)
		}
	} else {
		// Return all if no search criteria
		searchQuery = bleve.NewMatchAllQuery()
	}

	searchRequest := bleve.NewSearchRequest(searchQuery)
	searchRequest.Size = 100 // Limit results
	searchResult, err := ms.index.Search(searchRequest)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Fetch full memories from BoltDB
	var memories []Memory
	err = ms.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(memoryBucket))
		for _, hit := range searchResult.Hits {
			data := b.Get([]byte(hit.ID))
			if data == nil {
				continue
			}
			var memory Memory
			if err := json.Unmarshal(data, &memory); err != nil {
				continue
			}
			memories = append(memories, memory)
		}
		return nil
	})

	return memories, err
}

// Delete removes a memory
func (ms *MemoryStore) Delete(id string) error {
	// Delete from BoltDB
	err := ms.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(memoryBucket))
		return b.Delete([]byte(id))
	})
	if err != nil {
		return fmt.Errorf("failed to delete memory: %w", err)
	}

	// Delete from index
	if err := ms.index.Delete(id); err != nil {
		return fmt.Errorf("failed to delete from index: %w", err)
	}

	return nil
}

func main() {
	// Create memory store
	memStore, err := NewMemoryStore("memory.db")
	if err != nil {
		log.Fatalf("Failed to create memory store: %v", err)
	}
	defer memStore.Close()

	// Create MCP server
	s := server.NewMCPServer(
		"Simple Memory Server",
		"1.0.0",
		server.WithResourceCapabilities(true, true),
	)

	// Add memory tool
	addMemoryTool := mcp.NewTool("add_memory",
		mcp.WithDescription("Add a new memory with tags"),
		mcp.WithString("content",
			mcp.Required(),
			mcp.Description("The content to remember"),
		),
		mcp.WithString("type",
			mcp.Description("Type of memory: fact, conversation, reference"),
			mcp.Enum("fact", "conversation", "reference"),
		),
		mcp.WithArray("tags",
			mcp.DefaultString("string"),
			mcp.Description("Tags to categorize the memory"),
		),
	)

	s.AddTool(addMemoryTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		content, ok := request.Params.Arguments["content"].(string)
		if !ok {
			return mcp.NewToolResultError("content must be a string"), nil
		}

		memType := "fact"
		if t, ok := request.Params.Arguments["type"].(string); ok {
			memType = t
		}

		var tags []string
		if t, ok := request.Params.Arguments["tags"].([]interface{}); ok {
			for _, tag := range t {
				if strTag, ok := tag.(string); ok {
					tags = append(tags, strTag)
				}
			}
		}

		id, err := memStore.Add(content, memType, tags)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Memory stored with ID: %s", id)), nil
	})

	// Search memory tool
	searchTool := mcp.NewTool("search_memory",
		mcp.WithDescription("Search for memories by content or tags"),
		mcp.WithString("query",
			mcp.Description("Text to search for in memory content"),
		),
		mcp.WithArray("tags",
			mcp.DefaultString("string"),
			mcp.Description("Tags to filter by"),
		),
	)

	s.AddTool(searchTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query := ""
		if q, ok := request.Params.Arguments["query"].(string); ok {
			query = q
		}

		var tags []string
		if t, ok := request.Params.Arguments["tags"].([]interface{}); ok {
			for _, tag := range t {
				if strTag, ok := tag.(string); ok {
					tags = append(tags, strTag)
				}
			}
		}

		results, err := memStore.Search(query, tags)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		var response string
		for i, memory := range results {
			response += fmt.Sprintf("[%d] %s\n", i+1, memory.Content)
			response += fmt.Sprintf("   Type: %s, Tags: %v\n\n", memory.Type, memory.Tags)
		}

		if response == "" {
			response = "No matching memories found."
		}

		return mcp.NewToolResultText(response), nil
	})

	// Delete memory tool
	deleteTool := mcp.NewTool("delete_memory",
		mcp.WithDescription("Delete a memory by ID"),
		mcp.WithString("id",
			mcp.Required(),
			mcp.Description("ID of the memory to delete"),
		),
	)

	s.AddTool(deleteTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, ok := request.Params.Arguments["id"].(string)
		if !ok {
			return mcp.NewToolResultError("id must be a string"), nil
		}

		if err := memStore.Delete(id); err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Memory %s deleted successfully", id)), nil
	})

	// Start the server
	if err := server.ServeStdio(s); err != nil {
		fmt.Printf("Server error: %v\n", err)
	}
}
