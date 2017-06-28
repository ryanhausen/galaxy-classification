from __future__ import print_function,division

import datetime
import os
import argparse

parser = argparse.ArgumentParser(description='Post Helper')
parser.add_argument('title', type=str, help='Post title for display in double-quotes(\")')
parser.add_argument('--tags', help='Tags for the post')
parser.add_argument('--date', help='Date for the post in yyyy-mm-dd, optional')
parser.add_argument('--jekyll', help='Start jekyll server with drafts enabled after post is created.')
parser.add_argument('--editor', help='The editor command to open post after post has been created')

args = parser.parse_args()

date = datetime.date.today().isoformat()
tags = ''

hidden_title = args.title.replace(' ', '-').lower()
display_title = '"' + args.title + '"' 

if args.date:
    raise NotImplemented()

if args.tags:
    tags = args.tags

# make img dir
print('Making img dir..')
os.mkdir(f'./assets/imgs/{date}')

# make draft
print(f'Creating draft post {hidden_title}')

if '_drafts' not in os.listdir('.'):
    os.mkdir('_drafts')


with open(f'./_drafts/{date}-{hidden_title}.md', 'w') as f:
    mathjax = '{% include mathjax.html  %}\n'
    post_meta = '---\nlayout: default\ntitle: {}\ndate: {}\ncategories:{}\n---\n\n'.format(display_title,date,tags)
    f.write(post_meta + mathjax)
    
print('Finished.')

if args.editor:
    os.system(f'{args.editor} ./_drafts/{hidden_title}.md')
