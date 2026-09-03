package main

import (
	"fmt"
	"strings"
)

func addTask(tasks []string, title string) ([]string, error) {
	title = strings.TrimSpace(title)
	if title == "" {
		return tasks, fmt.Errorf("title is required")
	}
	return append(tasks, title), nil
}

func main() {
	tasks := []string{}
	var err error

	tasks, err = addTask(tasks, "Learn Go")
	if err != nil {
		fmt.Println("add task:", err)
		return
	}

	tasks, err = addTask(tasks, "Write HTTP API")
	if err != nil {
		fmt.Println("add task:", err)
		return
	}

	for index, title := range tasks {
		fmt.Printf("[%d] %s\n", index+1, title)
	}
}
