"""Generic embedding-based retrieval: embed, index, search.

Domain modules (skills, mcp) supply their own record types and use this
package for the shared infrastructure — embedder, index persistence, and
hybrid cosine + lexical retrieval.
"""
