# -*- coding: utf-8 -*-
"""
文档阅读器 - 读取和解析 Markdown 文档
"""

import os
from typing import Dict, List, Optional
import re


class DocumentReader:
    """文档阅读器类"""
    
    def __init__(self, doc_path: str):
        """
        初始化文档阅读器
        
        Args:
            doc_path: 文档路径
        """
        self.doc_path = doc_path
        self.content = ""
        self.sections = {}
        self.metadata = {}
        
        if os.path.exists(doc_path):
            self._load_document()
            self._parse_sections()
        else:
            raise FileNotFoundError(f"文档不存在: {doc_path}")
    
    def _load_document(self):
        """加载文档内容"""
        with open(self.doc_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
    
    def _parse_sections(self):
        """解析文档章节"""
        # 提取元数据（版本、日期等）
        version_match = re.search(r'\*\*版本\*\*:\s*(.+)', self.content)
        date_match = re.search(r'\*\*日期\*\*:\s*(.+)', self.content)
        
        if version_match:
            self.metadata['version'] = version_match.group(1).strip()
        if date_match:
            self.metadata['date'] = date_match.group(1).strip()
        
        # 按一级标题分割章节
        sections = re.split(r'\n## ', self.content)
        
        for section in sections[1:]:  # 跳过第一个（标题前的内容）
            lines = section.split('\n', 1)
            if len(lines) == 2:
                title = lines[0].strip()
                content = lines[1].strip()
                self.sections[title] = content
    
    def get_section(self, section_title: str) -> Optional[str]:
        """
        获取指定章节内容
        
        Args:
            section_title: 章节标题（支持部分匹配）
        
        Returns:
            章节内容，如果不存在返回 None
        """
        # 精确匹配
        if section_title in self.sections:
            return self.sections[section_title]
        
        # 部分匹配
        for title, content in self.sections.items():
            if section_title.lower() in title.lower():
                return content
        
        return None
    
    def list_sections(self) -> List[str]:
        """
        列出所有章节标题
        
        Returns:
            章节标题列表
        """
        return list(self.sections.keys())
    
    def get_full_content(self) -> str:
        """
        获取完整文档内容
        
        Returns:
            完整文档内容
        """
        return self.content
    
    def get_metadata(self) -> Dict[str, str]:
        """
        获取文档元数据
        
        Returns:
            元数据字典
        """
        return self.metadata
    
    def search(self, keyword: str, context_lines: int = 3) -> List[Dict[str, str]]:
        """
        在文档中搜索关键词
        
        Args:
            keyword: 搜索关键词
            context_lines: 上下文行数
        
        Returns:
            搜索结果列表 [{"section": "章节", "match": "匹配内容", "context": "上下文"}]
        """
        results = []
        
        for section_title, section_content in self.sections.items():
            lines = section_content.split('\n')
            
            for i, line in enumerate(lines):
                if keyword.lower() in line.lower():
                    # 提取上下文
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = '\n'.join(lines[start:end])
                    
                    results.append({
                        'section': section_title,
                        'match': line.strip(),
                        'context': context
                    })
        
        return results
    
    def get_summary(self, max_length: int = 500) -> str:
        """
        获取文档摘要
        
        Args:
            max_length: 最大长度
        
        Returns:
            文档摘要
        """
        # 提取第一段作为摘要
        lines = self.content.split('\n')
        summary_lines = []
        total_length = 0
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                summary_lines.append(line)
                total_length += len(line)
                if total_length >= max_length:
                    break
        
        return '\n'.join(summary_lines[:5])  # 最多5行


if __name__ == '__main__':
    # 测试
    doc_path = 'llm/doc/v0.0.1/Theoretical_principle_document.md'
    reader = DocumentReader(doc_path)
    
    print("=" * 60)
    print("文档元数据:")
    print(reader.get_metadata())
    
    print("\n" + "=" * 60)
    print("章节列表:")
    for i, section in enumerate(reader.list_sections(), 1):
        print(f"  {i}. {section}")
