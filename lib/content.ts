import fs from "fs"
import path from "path"
import matter from "gray-matter"
// Ensure all necessary types from your types/index.ts are imported
import type {
  Project,
  Post,
  HomeContent,
  AboutContent,
  FullProject,
  FullPost,
  ExperienceItem,
  EducationItem,
} from "@/types"
import yaml from "js-yaml"

const contentDirectory = path.join(process.cwd(), "content")

// Define the expected structure of the about/info.yaml file
interface AboutInfoYaml {
  name?: string // Use optional fields if they might be missing in YAML
  tagline?: string
  skills?: string[]
  experience?: ExperienceItem[]
  education?: EducationItem[]
  // Add any other fields expected in info.yaml
}

// Helper function to read files from a directory
function readDirectory(dir: string): string[] {
  const fullPath = path.join(contentDirectory, dir)
  try {
    // Check if directory exists before reading
    if (fs.existsSync(fullPath)) {
       return fs.readdirSync(fullPath);
    }
  } catch (error) {
      console.error(`Error reading directory: ${fullPath}`, error);
  }
  return []; // Return empty array if dir doesn't exist or on error
}


// Helper function to read a markdown file
function readMarkdownFile(filePath: string) {
  const fullPath = path.join(contentDirectory, filePath)
  if (!fs.existsSync(fullPath)) {
     console.warn(`Markdown file not found: ${fullPath}`);
    return null
  }
  try {
    const fileContents = fs.readFileSync(fullPath, "utf8")
    return matter(fileContents) // gray-matter handles parsing and data extraction
  } catch (error) {
      console.error(`Error reading or parsing markdown file: ${fullPath}`, error);
      return null;
  }
}

// Helper function to read a YAML file - returns 'unknown' for better type safety
function readYamlFile(filePath: string): unknown { // Return unknown
  const fullPath = path.join(contentDirectory, filePath)
  if (!fs.existsSync(fullPath)) {
    console.warn(`YAML file not found: ${fullPath}`);
    return null
  }
  try {
    const fileContents = fs.readFileSync(fullPath, "utf8")
    return yaml.load(fileContents)
  } catch (e) {
    console.error(`Error reading or parsing YAML file: ${fullPath}`, e)
    return null // Return null on error as well
  }
}

// Get home page content
export async function getHomeContent(): Promise<HomeContent> {
  const homeFile = readMarkdownFile("home/intro.md")

  const defaultContent: HomeContent = {
    title: "Hi, I'm Your Name",
    subtitle: "I'm a web developer specializing in building exceptional digital experiences.",
    description: "Based in Your Location, I'm passionate about creating intuitive and performant web applications.",
  }

  if (!homeFile?.data) { // Check if homeFile or its data property exists
    return defaultContent
  }

  return {
    title: homeFile.data.title || defaultContent.title,
    subtitle: homeFile.data.subtitle || defaultContent.subtitle,
    description: homeFile.data.description || defaultContent.description,
  }
}

// Get about page content (UPDATED LOGIC)
export async function getAboutContent(): Promise<AboutContent> {
  const infoFileRaw = readYamlFile("about/info.yaml") // Use 'unknown' type
  const storyFile = readMarkdownFile("about/story.md")

  const defaultBio = "<p>I'm a passionate web developer with over 5 years of experience building modern web applications.</p>"
  // Use bio from file if available, otherwise defaultBio
  const bioContent = storyFile?.content ?? defaultBio;

  // Default values for the entire AboutContent structure
  const defaultInfo: AboutContent = {
    name: "Your Name",
    tagline: "Web Developer & Designer",
    bio: bioContent, // Use bio determined above
    skills: ["JavaScript", "TypeScript", "React", "Next.js"],
    experience: [],
    education: [],
  }

  // Check if infoFileRaw is usable and assert its type
  // We check if it's an object and not null before asserting
  if (typeof infoFileRaw === "object" && infoFileRaw !== null) {
    // Assert the type AFTER confirming it's a non-null object
    const infoFile = infoFileRaw as AboutInfoYaml

    // Construct the result, falling back to defaults for each missing property
    return {
      name: infoFile.name ?? defaultInfo.name,
      tagline: infoFile.tagline ?? defaultInfo.tagline,
      bio: bioContent, // Bio is handled separately
      skills: infoFile.skills ?? defaultInfo.skills,
      experience: infoFile.experience ?? defaultInfo.experience,
      education: infoFile.education ?? defaultInfo.education,
    }
  } else {
    // If infoFileRaw is null, not an object, or yaml.load failed, return defaults
    console.warn("about/info.yaml not found or is invalid. Using default About content.")
    // Return the default structure, ensuring 'bio' is correctly included
    return defaultInfo;
  }
}


// Get all projects
export async function getAllProjects(): Promise<Project[]> {
  const projectFiles = readDirectory("projects")

  const projects = projectFiles.reduce<Project[]>((acc, file) => {
    if (file.endsWith(".md")) {
      const projectFile = readMarkdownFile(`projects/${file}`);
      if (projectFile?.data) {
        const slug = projectFile.data.slug || file.replace(/\.md$/, "");
        acc.push({
          title: projectFile.data.title ?? "Untitled Project",
          slug,
          date: projectFile.data.date ?? new Date().toISOString(),
          excerpt: projectFile.data.excerpt ?? "",
          coverImage: projectFile.data.coverImage || "/placeholder.svg?height=600&width=800",
          tags: projectFile.data.tags ?? [],
          githubLink: projectFile.data.githubLink ?? "#",
          liveDemoUrl: projectFile.data.liveDemoUrl ?? "#",
          category: projectFile.data.category ?? "Uncategorized",
          challenge: projectFile.data.challenge ?? "",
          solution: projectFile.data.solution ?? "",
          technologies: projectFile.data.technologies ?? [],
          features: projectFile.data.features ?? [],
          featured: projectFile.data.featured ?? false,
          lang: projectFile.data.lang || "en",
        });
      }
    }
    return acc;
  }, []);

  return projects.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

// Get all projects with content
export async function getAllProjectsWithContent(): Promise<FullProject[]> {
  const projectFiles = readDirectory("projects")

  const projects = projectFiles.reduce<FullProject[]>((acc, file) => {
    if (file.endsWith(".md")) {
        const projectFile = readMarkdownFile(`projects/${file}`);
        if (projectFile?.data && projectFile.content !== undefined && projectFile.content !== null) {
            const slug = projectFile.data.slug || file.replace(/\.md$/, "");
            acc.push({
                title: projectFile.data.title ?? "Untitled Project",
                slug,
                date: projectFile.data.date ?? new Date().toISOString(),
                excerpt: projectFile.data.excerpt ?? "",
                coverImage: projectFile.data.coverImage || "/placeholder.svg?height=600&width=800",
                tags: projectFile.data.tags ?? [],
                githubLink: projectFile.data.githubLink ?? "#",
                liveDemoUrl: projectFile.data.liveDemoUrl ?? "#",
                category: projectFile.data.category ?? "Uncategorized",
                challenge: projectFile.data.challenge ?? "",
                solution: projectFile.data.solution ?? "",
                technologies: projectFile.data.technologies ?? [],
                features: projectFile.data.features ?? [],
                content: projectFile.content,
                lang: projectFile.data.lang || "en",
                featured: projectFile.data.featured ?? false,
            });
        }
    }
    return acc;
  }, []);

  return projects.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

// New Get featured projects
// ===========================================

export async function getFeaturedProjects(count: number): Promise<Project[]> {
  const projects = await getAllProjects()
  return projects
    .filter((project) => project.featured === true)
    .slice(0, count)
}

// ===========================================

// Get a single project by slug
export async function getProjectBySlug(slug: string): Promise<FullProject | null> {
  const projects = await getAllProjectsWithContent()
  return projects.find((project) => project && project.slug === slug) || null
}

// Get related projects
export async function getRelatedProjects(project: Project, count: number): Promise<Project[]> {
   if (!project || !project.tags || !project.slug) {
      return [];
   }
  const allProjects = await getAllProjects()

  return allProjects
    .filter((p) => p && p.slug !== project.slug)
    .filter((p) => p.tags && p.tags.some((tag) => project.tags.includes(tag)))
    .slice(0, count)
}

// Get all blog posts
export async function getAllPosts(): Promise<Post[]> {
  const postFiles = readDirectory("blog")

  const posts = postFiles.reduce<Post[]>((acc, file) => {
    if (file.endsWith(".md")) {
        const postFile = readMarkdownFile(`blog/${file}`);
        if (postFile?.data) {
            const slug = postFile.data.slug || file.replace(/\.md$/, "");
            acc.push({
                title: postFile.data.title ?? "Untitled Post",
                slug,
                date: postFile.data.date ?? new Date().toISOString(),
                excerpt: postFile.data.excerpt ?? "",
                coverImage: postFile.data.coverImage || "/placeholder.svg?height=600&width=800",
                tags: postFile.data.tags ?? [],
                readTime: postFile.data.readTime || "5 min read",
                featured: postFile.data.featured ?? false,
            });
        }
    }
    return acc;
  }, []);

  return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

// Get all blog posts with content
export async function getAllPostsWithContent(): Promise<FullPost[]> {
  const postFiles = readDirectory("blog")

  const posts = postFiles.reduce<FullPost[]>((acc, file) => {
    if (file.endsWith(".md")) {
        const postFile = readMarkdownFile(`blog/${file}`);
        if (postFile?.data && postFile.content !== undefined && postFile.content !== null) {
            const slug = postFile.data.slug || file.replace(/\.md$/, "");
            acc.push({
                title: postFile.data.title ?? "Untitled Post",
                slug,
                date: postFile.data.date ?? new Date().toISOString(),
                excerpt: postFile.data.excerpt ?? "",
                coverImage: postFile.data.coverImage || "/placeholder.svg?height=600&width=800",
                tags: postFile.data.tags ?? [],
                readTime: postFile.data.readTime || "5 min read",
                content: postFile.content,
                featured: postFile.data.featured ?? false,
            });
        }
    }
    return acc;
  }, []);

  return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

// New Get Featured blog posts
// ===========================================

export async function getRecentPosts(count: number): Promise<Post[]> {
  const posts = await getAllPosts()
  return posts
    .filter((post) => post.featured === true)
    .slice(0, count)
}

// ===========================================


// Get a single blog post by slug
export async function getPostBySlug(slug: string): Promise<FullPost | null> {
  const posts = await getAllPostsWithContent()
   return posts.find((post) => post && post.slug === slug) || null
}

// Get related blog posts
export async function getRelatedPosts(post: Post, count: number): Promise<Post[]> {
  if (!post || !post.tags || !post.slug) {
      return [];
   }
  const allPosts = await getAllPosts()

  return allPosts
    .filter((p) => p && p.slug !== post.slug)
    .filter((p) => p.tags && p.tags.some((tag) => post.tags.includes(tag)))
    .slice(0, count)
}