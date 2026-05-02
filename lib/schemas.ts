// lib/schemas.ts
//
// Zod schemas for frontmatter validation. Used by lib/content.ts at load time
// so bad frontmatter fails loud during the build instead of corrupting pages
// silently.
//
// New fields beyond what the existing posts use (streamId, language, papers,
// generatedBy) are optional. They're populated by the Layer B publishing
// pipeline; manual posts ignore them.

import { z } from "zod"

// A flexible date string. Frontmatter dates are usually "YYYY-MM-DD" but we
// don't want to fight the user over zero-padding etc. — JS Date parsing is
// the real test. coerce.date() converts the string and refines on parse.
const dateString = z
  .string()
  .min(1, "date is required")
  .refine((s) => !Number.isNaN(Date.parse(s)), {
    message: "date must be a parseable date string (e.g., 2025-04-30)",
  })

const tags = z.array(z.string()).default([])

// Paper reference — populated by the AI-papers stream when a post is auto-
// generated from one or more arXiv / HF Daily papers.
export const PaperRef = z.object({
  title: z.string(),
  authors: z.array(z.string()).default([]),
  url: z.string().url(),
  arxivId: z.string().optional(),
})
export type PaperRef = z.infer<typeof PaperRef>

// Blog post frontmatter.
export const PostFrontmatter = z.object({
  title: z.string().min(1),
  slug: z.string().min(1).optional(), // falls back to filename in lib/content.ts
  date: dateString,
  tags,
  excerpt: z.string().default(""),
  readTime: z.string().default("5 min read"),
  featured: z.boolean().default(false),
  coverImage: z.string().optional(),

  // Pipeline-specific (all optional for backward compat with manual posts).
  streamId: z.string().optional(),
  language: z.enum(["en", "ar"]).default("en"),
  papers: z.array(PaperRef).optional(),
  generatedBy: z.string().optional(), // e.g. "claude-sonnet-4-6"
})
export type PostFrontmatter = z.infer<typeof PostFrontmatter>

// Project frontmatter (looser; existing project files have varied shapes).
export const ProjectFrontmatter = z.object({
  title: z.string().min(1),
  slug: z.string().min(1).optional(),
  date: dateString,
  tags,
  excerpt: z.string().default(""),
  coverImage: z.string().optional(),
  category: z.string().default("Uncategorized"),
  githubLink: z.string().optional(),
  liveDemoUrl: z.string().optional(),
  challenge: z.string().default(""),
  solution: z.string().default(""),
  technologies: z.array(z.string()).default([]),
  features: z.array(z.string()).default([]),
  featured: z.boolean().default(false),
  lang: z.string().default("en"),
})
export type ProjectFrontmatter = z.infer<typeof ProjectFrontmatter>

/**
 * Parse frontmatter with the given schema. Throws a helpful error including
 * the file path on validation failure so the build log points at the right
 * file.
 */
export function parseFrontmatter<S extends z.ZodTypeAny>(
  schema: S,
  data: unknown,
  filePath: string,
): z.infer<S> {
  const result = schema.safeParse(data)
  if (!result.success) {
    const issues = result.error.issues
      .map((i) => `  - ${i.path.join(".") || "(root)"}: ${i.message}`)
      .join("\n")
    throw new Error(
      `Invalid frontmatter in ${filePath}:\n${issues}`,
    )
  }
  return result.data
}
