import Image from "next/image"
import Link from "next/link"
import { Clock } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import type { Post } from "@/types"
import { formatDate } from "@/lib/utils"

interface BlogCardProps {
  post: Post
}

export default function BlogCard({ post }: BlogCardProps) {
  return (
    <Card className="overflow-hidden transition-all hover:shadow-md">
      <Link href={`/blog/${post.slug}`} className="block">
        <div className="relative aspect-video overflow-hidden">
          <Image
            src={post.coverImage || "/placeholder.svg"}
            alt={post.title}
            fill
            className="object-cover transition-transform hover:scale-105"
          />
        </div>
      </Link>

      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <time dateTime={post.date} className="text-sm text-muted-foreground">
            {formatDate(post.date)}
          </time>
          <div className="flex items-center text-sm text-muted-foreground">
            <Clock className="mr-1 h-3 w-3" />
            <span>{post.readTime}</span>
          </div>
        </div>

        <Link href={`/blog/${post.slug}`} className="block">
          <h3 className="text-xl font-bold mb-2 hover:underline">{post.title}</h3>
        </Link>

        <p className="text-muted-foreground line-clamp-2 mb-4">{post.excerpt}</p>

        <div className="flex flex-wrap gap-2">
          {post.tags.slice(0, 3).map((tag) => (
            <Badge key={tag} variant="secondary" className="text-xs">
              {tag}
            </Badge>
          ))}
          {post.tags.length > 3 && (
            <Badge variant="outline" className="text-xs">
              +{post.tags.length - 3} more
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
