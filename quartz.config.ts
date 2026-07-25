import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "markedown_",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "quartz.jzhao.xyz",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",

    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "EB Garamond", // Use Lato for headers
        body: "EB Garamond",   // Use Lato for the body
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#FFFBDE",      // Page background
          lightgray: "#e5e5e5",   // Borders
          gray: "#b8b8b8",       // Secondary text (e.g., dates)
          darkgray: "#000000",   // Body text
          dark: "#000000",       // Headings
          secondary: "#17a2b8",   // Link color from your personal site
          tertiary: "#f09228",    // Link hover color from your personal site
          highlight: "#FFFBDE", // A more subtle highlight
          textHighlight: "#f0922855", 
        },
        darkMode: {
          light: "#1c1c1c",
          lightgray: "#393639",
          gray: "#646464",
          darkgray: "#d4d4d4",
          dark: "#ebebec",
          secondary: "#579cd9",   // Lighter blue for dark mode links
          tertiary: "#ffb86c",    // Lighter orange for dark mode hover
          highlight: "1c1c1c",
          textHighlight: "#ffb86c55",
        },
      },
    },

  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
    //   Plugin.CustomOgImages(),
    ],
  },
}

export default config
