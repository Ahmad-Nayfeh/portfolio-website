// types/index.ts

export interface Project {
  title: string
  slug: string
  date: string
  excerpt: string
  coverImage: string
  tags: string[]
  githubLink: string
  liveDemoUrl: string
  category: string
  challenge: string
  solution: string
  technologies: string[]
  features: string[]
  featured?: boolean
  lang?: string
  status?: string
  gallery: string[]
  highlights: string[]
}

export interface FullProject extends Project {
  content: string
}

export interface Post {
  title: string
  slug: string
  date: string
  excerpt: string
  coverImage: string
  tags: string[]
  readTime: string
  featured?: boolean
}

export interface FullPost extends Post {
  content: string
}

export interface NowItem {
  label: string
  detail: string
}

export interface HomeContent {
  title: string
  subtitle: string
  description: string
  now: NowItem[]
}

export interface WorkStep {
  title: string
  detail: string
}

export interface WorkContent {
  title: string
  intro: string
  products: string[]
  steps: WorkStep[]
  automation: string
}

export interface SkillGroup {
  name: string
  items: string[]
}

export interface AboutContent {
  name: string
  tagline: string
  bio: string
  skills: string[]
  skillGroups: SkillGroup[]
  languages: string[]
  experience: ExperienceItem[]
  education: EducationItem[]
}

export interface ExperienceItem {
  title: string
  company: string
  location: string
  startDate: string
  endDate: string
  description: string
}

export interface EducationItem {
  degree: string
  institution: string
  location: string
  startDate: string
  endDate: string
  description: string
}