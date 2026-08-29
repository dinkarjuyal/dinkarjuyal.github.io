source "https://rubygems.org"

gem "jekyll"

# Core plugins that directly affect site building
group :jekyll_plugins do
  gem "jekyll-3rd-party-libraries"
  gem "jekyll-archives-v2"
  gem "jekyll-cache-bust"
  gem "jekyll-feed"
  gem "jekyll-link-attributes"
  gem "jekyll-paginate-v2"
  gem "jekyll-regex-replace"
  gem "jekyll-scholar"
  gem "jekyll-sitemap"
  gem "jekyll-socials"
  gem "jekyll-toc"

  gem "classifier-reborn" # used for related-posts similarity
end

# Runtime deps of the plugins above (outside :jekyll_plugins)
group :other_plugins do
  gem "observer" # used by jekyll-scholar
end

gem "webrick", "~> 1.7"
